"""Dense vs BM25 vs fusion: does lexical scoring fix the antonym failure?

THE TARGET

`when does registration begin` returned *Registration Ends — 1 October 2026* at
rank 1, with *Registration Begins — 7 May 2026* at rank 2. Worse, both passages
land in the context window, so the groundedness check judges "Registration
begins on 1 October 2026" fully supported and passes it (bench/guardrail_cases
g020). A ranking bug and a guardrail blind spot compose into a confidently wrong
answer to the likeliest question this corpus will ever get. Fixing the guardrail
cannot help; the ranking has to change.

WHAT IS MEASURED

  quality   recall@5, precision@1, MRR over the answerable queries, for dense
            alone, BM25 alone, RRF fusion, and a dense/lexical weight sweep.

  stability A bootstrap CI on precision@1. With 37 answerable queries the gap
            between two fusion settings can be two queries wide, and picking
            the peak of a sweep evaluated on the same set it was tuned on is
            how you ship a number that does not survive contact with new
            queries. The CI is here to stop that.

  cost      Added latency. Retrieval latency is the one performance claim this
            project actually makes, so a quality win that blows the budget is
            not a win.

Note on the score gate: fusion scores are rank-based or min-max normalised and
carry NO calibrated scale, so they must never feed check_score. Retriever keeps
reporting the dense cosine for gating and uses fusion only for ordering. See
app/rag/retrieve.py.

Run:  py -m bench.bench_hybrid
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from app.index.build import index_path
from app.index.embed import DEFAULT_MODEL, Embedder, model_slug
from app.index.lexical import BM25
from app.index.store import VectorStore

QUERIES = Path("bench/queries.yaml")
REPORT_DIR = Path("bench/reports")


def load_queries() -> list[dict]:
    qs = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))
    for q in qs:
        q.setdefault("gold", [])
    return qs


def ranks_from(scores: np.ndarray) -> np.ndarray:
    """1-based dense ranking (rank 1 = best)."""
    order = np.argsort(-scores)
    r = np.empty(len(scores), dtype=np.int32)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare dense, BM25 and fusion rankers.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strategy", default="parent_child")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=40, help="latency repeats per query")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    queries = load_queries()
    answerable = [q for q in queries if q["expect"] == "answer"]

    embedder = Embedder(model_name=args.model)
    store = VectorStore.load(index_path(args.model, "flat", args.strategy))
    payloads = store.payloads
    doc_of = [p["doc_id"] for p in payloads]
    bm25 = BM25([p["embed_text"] for p in payloads])

    print(f"queries : {len(queries)} ({len(answerable)} answerable)")
    print(f"index   : {len(store)} chunks, strategy={args.strategy}\n")

    def collapse(order: list[int]) -> list[str]:
        """Chunk ranking -> unique parent doc_ids, mirroring Retriever."""
        seen: set[str] = set()
        out: list[str] = []
        for i in order:
            d = doc_of[i]
            if d in seen:
                continue
            seen.add(d)
            out.append(d)
            if len(out) >= args.k:
                break
        return out

    # Dense scores over every chunk, so fusion sees the same candidate set as
    # BM25. At 91 vectors a full scan is the same thing FAISS flat already does.
    matrix = store.vectors

    rankers: dict[str, object] = {}

    def dense_scores(q: str) -> np.ndarray:
        return matrix @ embedder.embed_query(q)

    rankers["dense only"] = lambda q: collapse(list(np.argsort(-dense_scores(q))))
    rankers["bm25 only"] = lambda q: collapse(list(np.argsort(-bm25.scores(q))))

    def make_rrf(kk: int):
        def f(q: str) -> list[str]:
            fused = 1.0 / (kk + ranks_from(dense_scores(q))) + 1.0 / (
                kk + ranks_from(bm25.scores(q))
            )
            return collapse(list(np.argsort(-fused)))

        return f

    def make_alpha(a: float):
        def f(q: str) -> list[str]:
            fused = a * minmax(dense_scores(q)) + (1 - a) * minmax(bm25.scores(q))
            return collapse(list(np.argsort(-fused)))

        return f

    for kk in (10, 60):
        rankers[f"rrf k={kk}"] = make_rrf(kk)
    for a in (0.9, 0.8, 0.7, 0.6, 0.5, 0.3):
        rankers[f"alpha={a}"] = make_alpha(a)

    # --- quality ------------------------------------------------------------
    results: dict[str, dict] = {}
    for name, fn in rankers.items():
        per_query: list[dict] = []
        got_all: dict[str, list[str]] = {}
        for q in queries:
            got = fn(q["query"])
            got_all[q["id"]] = got
            if q["expect"] != "answer":
                continue
            gold = set(q["gold"])
            rank = next((i + 1 for i, d in enumerate(got) if d in gold), 0)
            per_query.append(
                {
                    "id": q["id"],
                    "recall": int(bool(set(got) & gold)),
                    "p1": int(got[0] in gold),
                    "rr": 1.0 / rank if rank else 0.0,
                }
            )
        n = len(per_query)
        results[name] = {
            "recall@5": sum(r["recall"] for r in per_query) / n,
            "prec@1": sum(r["p1"] for r in per_query) / n,
            "mrr": sum(r["rr"] for r in per_query) / n,
            "per_query": per_query,
            "got": got_all,
        }

    # --- bootstrap CI on precision@1 ----------------------------------------
    rng = random.Random(0)
    n = len(answerable)
    idx_samples = [[rng.randrange(n) for _ in range(n)] for _ in range(args.boot)]
    for name, r in results.items():
        p1 = [q["p1"] for q in r["per_query"]]
        boots = sorted(sum(p1[i] for i in s) / n for s in idx_samples)
        r["p1_ci95"] = (
            round(boots[int(0.025 * args.boot)], 3),
            round(boots[int(0.975 * args.boot)], 3),
        )

    base = results["dense only"]
    print(f"{'ranker':<16}{'recall@5':>9}{'prec@1':>8}{'  95% CI':<16}{'mrr':>7}   vs dense")
    for name, r in results.items():
        d = r["prec@1"] - base["prec@1"]
        delta = "  --  " if name == "dense only" else f"{d:+.3f}"
        lo, hi = r["p1_ci95"]
        print(
            f"{name:<16}{r['recall@5']:>9.3f}{r['prec@1']:>8.3f}"
            f"  [{lo:.3f},{hi:.3f}]  {r['mrr']:>7.3f}   {delta}"
        )

    # Paired comparison: same queries, so count who wins where rather than
    # comparing two averages that share most of their inputs.
    print("\npaired vs dense (queries fixed / broken at rank 1):")
    for name, r in results.items():
        if name == "dense only":
            continue
        fixed = sum(
            1 for a, b in zip(base["per_query"], r["per_query"]) if not a["p1"] and b["p1"]
        )
        broke = sum(
            1 for a, b in zip(base["per_query"], r["per_query"]) if a["p1"] and not b["p1"]
        )
        print(f"  {name:<16} +{fixed} fixed  -{broke} broken   net {fixed - broke:+d}")

    # --- the target case ----------------------------------------------------
    print("\nq001 'when does registration begin'  (gold timeline-00):")
    for name, r in results.items():
        got = r["got"]["q001"]
        mark = "FIXED" if got[0] == "timeline-00" else "     "
        print(f"  {name:<16}{mark}  {got[:3]}")

    print("\nq036 'how many builders attend'  (needs about-00 AND timeline-09 to cite the conflict):")
    for name, r in results.items():
        got = r["got"]["q036"]
        mark = "BOTH " if {"about-00", "timeline-09"} <= set(got) else "     "
        print(f"  {name:<16}{mark}  {got[:3]}")

    # --- latency ------------------------------------------------------------
    print("\nlatency (ms per query, warm):")
    timings: dict[str, dict] = {}
    for name in ("dense only", "rrf k=60"):
        fn = rankers[name]
        samples: list[float] = []
        for _ in range(args.repeats):
            for q in queries:
                t0 = time.perf_counter()
                fn(q["query"])
                samples.append((time.perf_counter() - t0) * 1000)
        s = sorted(samples)
        timings[name] = {
            "p50": round(s[len(s) // 2], 3),
            "p100": round(s[-1], 3),
            "mean": round(statistics.mean(s), 3),
            "n": len(s),
        }
        t = timings[name]
        print(f"  {name:<16} p50 {t['p50']:>6.3f}  p100 {t['p100']:>7.3f}  "
              f"mean {t['mean']:>6.3f}  ({t['n']} calls)")

    lex_only: list[float] = []
    for _ in range(args.repeats):
        for q in queries:
            t0 = time.perf_counter()
            bm25.scores(q["query"])
            lex_only.append((time.perf_counter() - t0) * 1000)
    lex_only.sort()
    print(f"  {'bm25 scoring':<16} p50 {lex_only[len(lex_only)//2]:>6.3f}  "
          f"p100 {lex_only[-1]:>7.3f}  mean {statistics.mean(lex_only):>6.3f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"hybrid-{model_slug(args.model)}-{args.strategy}.json"
    out.write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model,
                    "strategy": args.strategy,
                    "k": args.k,
                    "bootstrap": args.boot,
                },
                "n_answerable": n,
                "rankers": {
                    name: {
                        "recall@5": round(r["recall@5"], 4),
                        "prec@1": round(r["prec@1"], 4),
                        "p1_ci95": r["p1_ci95"],
                        "mrr": round(r["mrr"], 4),
                    }
                    for name, r in results.items()
                },
                "latency_ms": timings,
                "bm25_only_ms": {
                    "p50": round(lex_only[len(lex_only) // 2], 3),
                    "p100": round(lex_only[-1], 3),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
