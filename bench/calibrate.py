"""Calibrate the retrieval guardrail -- and test whether it can work at all.

The retrieval benchmark showed that absolute top-1 similarity CANNOT separate
answerable queries from must-refuse ones on this corpus: the populations
overlap (answerable min 0.8207 < refuse max 0.8655), so every possible cutoff
either lets hallucination-bait through or refuses real questions.

bge-family models are known to compress cosine similarities into a narrow high
band, which makes an absolute cutoff especially fragile. So instead of tuning a
number that cannot work, this script asks a different question: is there a
*relative* signal that does?

Candidate gating signals
------------------------
top1          absolute top-1 score (the naive baseline, expected to fail)
margin_12     top1 - top2          -- is the best match distinctly best?
margin_1k     top1 - mean(rest)    -- does top1 stand out from the field?
ratio_1k      top1 / mean(rest)    -- scale-free version of the above
spread        max - min over top-k -- diffuse matches indicate no real hit

The intuition for the relative signals: when a query genuinely matches one
chunk, that chunk should stand out. When a query matches nothing (the capital
of France), every chunk is equally mediocre and the top-k scores bunch
together. Absolute height says little; distinctness says a lot.

Each signal is scored by the best accuracy any cutoff achieves, plus AUC, which
is threshold-independent and so a fairer measure of whether the signal carries
information at all.

Run:  py -m bench.calibrate
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

from app.index.build import index_path
from app.index.embed import DEFAULT_MODEL, Embedder, model_slug
from app.index.store import VectorStore

QUERIES = Path("bench/queries.yaml")
REPORT_DIR = Path("bench/reports")


def signals(scores: list[float]) -> dict[str, float]:
    """Derive every candidate gating signal from one query's top-k scores."""
    if not scores:
        return {"top1": 0.0, "margin_12": 0.0, "margin_1k": 0.0, "ratio_1k": 0.0, "spread": 0.0}
    top1 = scores[0]
    rest = scores[1:] or [top1]
    mean_rest = statistics.mean(rest)
    return {
        "top1": top1,
        "margin_12": top1 - rest[0],
        "margin_1k": top1 - mean_rest,
        "ratio_1k": top1 / mean_rest if mean_rest else 0.0,
        "spread": max(scores) - min(scores),
    }


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random answerable query scores above a random refuse one.

    0.5 means the signal is noise; 1.0 means perfect separation. Computed by
    direct pair counting -- n is small enough that exactness beats cleverness.
    """
    if not pos or not neg:
        return 0.0
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_cutoff(pos: list[float], neg: list[float]) -> dict:
    """Sweep every midpoint between observed values; keep the best accuracy."""
    vals = sorted(set(pos + neg))
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or vals
    best: dict = {"accuracy": -1.0}
    for t in cands:
        kept = sum(1 for p in pos if p >= t)
        blocked = sum(1 for n in neg if n < t)
        acc = (kept + blocked) / (len(pos) + len(neg))
        if acc > best["accuracy"]:
            best = {
                "cutoff": round(t, 4),
                "accuracy": round(acc, 4),
                "answerable_kept": f"{kept}/{len(pos)}",
                "refuse_blocked": f"{blocked}/{len(neg)}",
                "false_refusals": len(pos) - kept,
                "hallucination_risk": len(neg) - blocked,
            }
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the retrieval guardrail.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strategy", default="parent_child")
    ap.add_argument("--kind", default="flat")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    queries = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))
    embedder = Embedder(model_name=args.model)
    store = VectorStore.load(index_path(args.model, args.kind, args.strategy))

    per_query: list[dict] = []
    for q in queries:
        results = store.search(embedder.embed_query(q["query"]), k=args.k)
        sc = [s for s, _ in results]
        per_query.append(
            {
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "expect": q["expect"],
                "scores": [round(s, 4) for s in sc],
                "signals": {k: round(v, 4) for k, v in signals(sc).items()},
            }
        )

    names = ["top1", "margin_12", "margin_1k", "ratio_1k", "spread"]
    print(f"model={args.model} strategy={args.strategy} k={args.k}")
    print(f"queries: {len(queries)}\n")
    print(f"{'signal':<12} {'AUC':>6}  {'best acc':>8}  {'cutoff':>8}  "
          f"{'kept':>7}  {'blocked':>8}  {'halluc':>6}")
    print("-" * 68)

    table: dict[str, dict] = {}
    for name in names:
        pos = [r["signals"][name] for r in per_query if r["expect"] == "answer"]
        neg = [r["signals"][name] for r in per_query if r["expect"] == "refuse"]
        a = auc(pos, neg)
        b = best_cutoff(pos, neg)
        table[name] = {"auc": round(a, 4), **b}
        print(f"{name:<12} {a:>6.3f}  {b['accuracy']:>8.3f}  {b['cutoff']:>8.4f}  "
              f"{b['answerable_kept']:>7}  {b['refuse_blocked']:>8}  "
              f"{b['hallucination_risk']:>6}")

    winner = max(names, key=lambda n: (table[n]["auc"], table[n]["accuracy"]))
    print(f"\nbest signal: {winner} (AUC {table[winner]['auc']:.3f})")

    # Accuracy is the WRONG objective for this gate. A false refusal annoys the
    # user; a hallucination destroys trust in the whole system. They are not
    # symmetric, so instead of one "best" cutoff we show the tradeoff curve and
    # let the operating point be an explicit decision.
    pos = [r["signals"][winner] for r in per_query if r["expect"] == "answer"]
    neg = [r["signals"][winner] for r in per_query if r["expect"] == "refuse"]

    print(f"\noperating points for {winner} (cost of blocking more refusals):")
    print(f"  {'cutoff':>8}  {'answerable kept':>16}  {'refuse blocked':>15}  {'leaks':>6}")
    vals = sorted(set(pos + neg))
    shown: set[tuple[int, int]] = set()
    operating: list[dict] = []
    for t in [(a + b) / 2 for a, b in zip(vals, vals[1:])]:
        kept = sum(1 for p in pos if p >= t)
        blocked = sum(1 for n in neg if n < t)
        if (kept, blocked) in shown:
            continue
        shown.add((kept, blocked))
        row = {
            "cutoff": round(t, 4),
            "kept": kept,
            "of_answerable": len(pos),
            "blocked": blocked,
            "of_refuse": len(neg),
            "leaks": len(neg) - blocked,
        }
        operating.append(row)
        # Only print rows where we still keep most real questions.
        if kept >= len(pos) * 0.85:
            print(f"  {t:>8.4f}  {f'{kept}/{len(pos)}':>16}  "
                  f"{f'{blocked}/{len(neg)}':>15}  {len(neg) - blocked:>6}")

    # The real question is not "what is the best cutoff" but "can ANY cutoff
    # make this gate sufficient on its own". Judge that by leakage at a cutoff
    # that does not sacrifice real questions -- not by AUC.
    safe = [r for r in operating if r["kept"] >= len(pos) * 0.95]
    min_leaks = min((r["leaks"] for r in safe), default=len(neg))
    print(
        f"\nAt any cutoff keeping >=95% of answerable queries, the best achievable\n"
        f"leakage is {min_leaks}/{len(neg)} must-refuse queries getting through."
    )
    if min_leaks > 1:
        print(
            "\nCONCLUSION: the score gate CANNOT be the guardrail. It is a cheap\n"
            "first pass that avoids wasting an LLM call on obvious misses, but a\n"
            "groundedness check on the generated answer is mandatory -- it is the\n"
            "only thing standing between these leaked queries and a fabricated\n"
            "answer. Design accordingly."
        )

    # Name the leaks explicitly: this is the acceptance test for groundedness.
    cut = table[winner]["cutoff"]
    leaks = [r for r in per_query if r["expect"] == "refuse" and r["signals"][winner] >= cut]
    if leaks:
        print(f"\nrefusals slipping past {winner} >= {cut} "
              f"(groundedness check must catch all of these):")
        for r in leaks:
            print(f"  [{r['category']:<22}] {r['query']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"calibration-{model_slug(args.model)}-{args.strategy}.json"
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "signals": table,
                "winner": winner,
                "operating_points": operating,
                "min_leaks_at_95pct_recall": min_leaks,
                "leaked_queries": [
                    {"id": r["id"], "query": r["query"], "category": r["category"]}
                    for r in leaks
                ],
                "per_query": per_query,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
