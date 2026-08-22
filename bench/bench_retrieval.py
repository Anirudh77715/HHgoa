"""Retrieval benchmark: latency distribution + recall + threshold calibration.

Three things are measured, because any one alone is misleading:

  latency   p50/p70/p100 per stage. Reported per stage rather than as a single
            number so the honest claim ("retrieval is sub-Xms") is separable
            from generation, which is a network call to a free-tier API and
            will never be sub-200ms.

  recall@k  Whether a gold doc actually reached top-k. An index that returns
            garbage in 2ms is fast and worthless; latency without recall is a
            vanity metric.

  prec@1    Whether the FIRST result is a gold doc. recall@5 alone hid a real
            bug: a 41-char empty-state doc ranked #1 for unrelated queries
            while the gold doc still sat somewhere in the top 5, so recall@5
            stayed high and the failure stayed invisible. The `attractors`
            block names any doc that keeps winning rank 1 without being gold,
            so that class of bug cannot hide behind an averaged metric again.

  threshold Calibrates RETRIEVAL_MIN_SCORE from data instead of guessing. We
            sweep candidate cutoffs and report which one best separates
            answerable queries from ones that must be refused. The guardrail
            is only as good as this number, and a guessed number is how RAG
            systems end up confidently answering questions they should decline.

The whole benchmark is local and free -- no API key, no rate limit -- which is
why it can afford enough repeats for p100 to mean something rather than being
one unlucky sample.

Run:  py -m bench.bench_retrieval
      py -m bench.bench_retrieval --repeats 200 --k 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from app.index.build import index_path
from app.index.embed import DEFAULT_MODEL, Embedder, model_slug
from app.index.store import VectorStore

QUERIES = Path("bench/queries.yaml")
REPORT_DIR = Path("bench/reports")
STRATEGIES = ("structural", "recursive", "parent_child")

# A doc that wins rank 1 for this many queries it is not gold for is reported
# as an attractor. 3 of 48 is already well past what topic overlap explains.
ATTRACTOR_MIN = 3


def pct(vals: list[float], p: float) -> float:
    """Nearest-rank percentile. p100 is the true max, i.e. the worst case a
    user actually hit -- not an interpolated estimate that hides it."""
    if not vals:
        return 0.0
    s = sorted(vals)
    if p >= 100:
        return s[-1]
    idx = min(len(s) - 1, int(round((p / 100) * len(s) + 0.5)) - 1)
    return s[max(0, idx)]


def fmt(vals: list[float]) -> str:
    return (
        f"p50 {pct(vals, 50):6.2f}  p70 {pct(vals, 70):6.2f}  "
        f"p100 {pct(vals, 100):6.2f}  mean {statistics.mean(vals):6.2f}"
    )


def load_queries() -> list[dict]:
    qs = yaml.safe_load(QUERIES.read_text(encoding="utf-8"))
    for q in qs:
        q.setdefault("gold", [])
        if q["expect"] == "answer" and not q["gold"]:
            raise ValueError(f"{q['id']}: expect=answer requires gold docs")
        if q["expect"] == "refuse" and q["gold"]:
            raise ValueError(f"{q['id']}: expect=refuse must have empty gold")
    return qs


def bench_strategy(
    name: str, kind: str, embedder: Embedder, queries: list[dict], repeats: int, k: int
) -> dict:
    store = VectorStore.load(index_path(embedder.model_name, kind, name))

    embed_ms: list[float] = []
    search_ms: list[float] = []
    total_ms: list[float] = []

    # Correctness pass: deterministic, so once is enough.
    hits = 0
    p1_hits = 0
    rr_sum = 0.0
    answerable = 0
    by_cat: dict[str, list[int]] = defaultdict(list)
    p1_by_cat: dict[str, list[int]] = defaultdict(list)
    top_scores: dict[str, float] = {}
    detail: list[dict] = []
    # Every query's rank-1 doc, and the subset where that doc was not gold.
    # A doc that dominates the second counter is an attractor, not a match.
    top1_all: Counter[str] = Counter()
    top1_wrong: Counter[str] = Counter()

    for q in queries:
        qv = embedder.embed_query(q["query"])
        results = store.search(qv, k=k)
        got_docs = [p["doc_id"] for _, p in results]
        top = results[0][0] if results else 0.0
        top_scores[q["id"]] = top

        gold = set(q["gold"])
        top1 = got_docs[0] if got_docs else ""
        # Rank counts retrieved rows. Under parent_child several rows can share
        # a parent, so this is a lower bound on the rank a caller sees after
        # Retriever collapses to unique parents -- never an overstatement.
        rank = next((i + 1 for i, d in enumerate(got_docs) if d in gold), 0)
        p1 = (top1 in gold) if gold else None

        if top1:
            top1_all[top1] += 1
            if top1 not in gold:
                top1_wrong[top1] += 1

        hit = bool(set(got_docs) & gold) if gold else None
        if q["expect"] == "answer":
            answerable += 1
            hits += int(bool(hit))
            p1_hits += int(bool(p1))
            rr_sum += 1.0 / rank if rank else 0.0
            by_cat[q["category"]].append(int(bool(hit)))
            p1_by_cat[q["category"]].append(int(bool(p1)))

        detail.append(
            {
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "expect": q["expect"],
                "top_score": round(top, 4),
                "hit": hit,
                "p1": p1,
                "rank": rank,
                "top_docs": got_docs[:3],
            }
        )

    # Latency pass: warm, repeated, unmeasured setup already done.
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            qv = embedder.embed_query(q["query"])
            t1 = time.perf_counter()
            store.search(qv, k=k)
            t2 = time.perf_counter()
            embed_ms.append((t1 - t0) * 1000)
            search_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)

    return {
        "strategy": name,
        "kind": kind,
        "n_vectors": len(store),
        "n_calls": len(total_ms),
        "recall_at_k": round(hits / answerable, 4) if answerable else 0.0,
        "precision_at_1": round(p1_hits / answerable, 4) if answerable else 0.0,
        "mrr_at_k": round(rr_sum / answerable, 4) if answerable else 0.0,
        "hits": hits,
        "p1_hits": p1_hits,
        "answerable": answerable,
        "recall_by_category": {
            c: round(sum(v) / len(v), 3) for c, v in sorted(by_cat.items())
        },
        "precision_at_1_by_category": {
            c: round(sum(v) / len(v), 3) for c, v in sorted(p1_by_cat.items())
        },
        "attractors": [
            {
                "doc_id": d,
                "rank1_total": top1_all[d],
                "rank1_not_gold": n,
                "share_of_queries": round(n / len(queries), 3),
            }
            for d, n in top1_wrong.most_common()
            if n >= ATTRACTOR_MIN
        ],
        "latency_ms": {
            "embed": {p: round(pct(embed_ms, p), 3) for p in (50, 70, 100)},
            "search": {p: round(pct(search_ms, p), 3) for p in (50, 70, 100)},
            "total": {p: round(pct(total_ms, p), 3) for p in (50, 70, 100)},
            "embed_mean": round(statistics.mean(embed_ms), 3),
            "search_mean": round(statistics.mean(search_ms), 3),
            "total_mean": round(statistics.mean(total_ms), 3),
        },
        "_raw": {"embed": embed_ms, "search": search_ms, "total": total_ms},
        "top_scores": top_scores,
        "detail": detail,
    }


def calibrate(queries: list[dict], top_scores: dict[str, float]) -> dict:
    """Find the cutoff that best separates answerable from must-refuse.

    Reported with the margin between the two populations: a wide margin means
    the threshold is robust, a narrow one means the guardrail is fragile and a
    groundedness check has to carry the weight instead.
    """
    ans = [top_scores[q["id"]] for q in queries if q["expect"] == "answer"]
    ref = [top_scores[q["id"]] for q in queries if q["expect"] == "refuse"]
    if not ans or not ref:
        return {}

    best = None
    for i in range(20, 96):
        t = i / 100
        kept = sum(1 for s in ans if s >= t)      # answerable, correctly kept
        blocked = sum(1 for s in ref if s < t)    # must-refuse, correctly blocked
        acc = (kept + blocked) / (len(ans) + len(ref))
        if best is None or acc > best["accuracy"]:
            best = {
                "threshold": round(t, 2),
                "accuracy": round(acc, 4),
                "answerable_kept": f"{kept}/{len(ans)}",
                "refuse_blocked": f"{blocked}/{len(ref)}",
            }

    return {
        "best": best,
        "answerable_scores": {
            "min": round(min(ans), 4),
            "p10": round(pct(ans, 10), 4),
            "median": round(statistics.median(ans), 4),
        },
        "refuse_scores": {
            "median": round(statistics.median(ref), 4),
            "p90": round(pct(ref, 90), 4),
            "max": round(max(ref), 4),
        },
        # Positive margin = clean separation. Negative = the populations
        # overlap and NO single threshold can separate them.
        "separation_margin": round(min(ans) - max(ref), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark retrieval per chunking strategy.")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--kind", choices=("flat", "hnsw"), default="flat")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    queries = load_queries()
    n_ans = sum(1 for q in queries if q["expect"] == "answer")
    print(f"queries: {len(queries)}  ({n_ans} answerable, {len(queries) - n_ans} must-refuse)")
    print(f"model  : {args.model}")
    print(f"repeats: {args.repeats}  k={args.k}  index={args.kind}\n")

    embedder = Embedder(model_name=args.model)  # __init__ warms the ONNX graph

    results = []
    for name in STRATEGIES:
        path = index_path(args.model, args.kind, name)
        if not path.exists():
            print(f"skip {name}: {path} missing — run `py -m app.index.build`")
            continue
        r = bench_strategy(name, args.kind, embedder, queries, args.repeats, args.k)
        r["calibration"] = calibrate(queries, r["top_scores"])
        results.append(r)

        lat = r["latency_ms"]
        print(f"=== {name} ({r['n_vectors']} vectors, {r['n_calls']} timed calls) ===")
        print(f"  recall@{args.k}   : {r['recall_at_k']:.3f}  ({r['hits']}/{r['answerable']})")
        print(f"  prec@1     : {r['precision_at_1']:.3f}  "
              f"({r['p1_hits']}/{r['answerable']})   mrr@{args.k} {r['mrr_at_k']:.3f}")
        print(f"  embed  ms  : {fmt(r['_raw']['embed'])}")
        print(f"  search ms  : {fmt(r['_raw']['search'])}")
        print(f"  TOTAL  ms  : {fmt(r['_raw']['total'])}")
        cal = r["calibration"]
        if cal:
            b = cal["best"]
            print(f"  threshold  : {b['threshold']} (acc {b['accuracy']:.3f}, "
                  f"kept {b['answerable_kept']}, blocked {b['refuse_blocked']})")
            print(f"  separation : {cal['separation_margin']:+.4f} "
                  f"(answerable min {cal['answerable_scores']['min']}, "
                  f"refuse max {cal['refuse_scores']['max']})")
        print("  recall / prec@1 by category:")
        p1c = r["precision_at_1_by_category"]
        for c, v in r["recall_by_category"].items():
            flag = "  <-- weak" if v < 0.8 else ""
            print(f"      {c:<24} r {v:.2f}   p1 {p1c.get(c, 0.0):.2f}{flag}")
        if r["attractors"]:
            print("  ATTRACTORS (won rank 1 while not gold):")
            for a in r["attractors"]:
                print(f"      {a['doc_id']:<20} {a['rank1_not_gold']:>3} queries "
                      f"({a['share_of_queries']:.0%})")
        print()

    if not results:
        print("no indexes found")
        return 1

    best = max(results, key=lambda r: (r["recall_at_k"], -r["latency_ms"]["total"][50]))
    print(f"best recall@{args.k}: {best['strategy']} "
          f"({best['recall_at_k']:.3f} @ p50 {best['latency_ms']['total'][50]}ms)")
    best_p1 = max(results, key=lambda r: (r["precision_at_1"], -r["latency_ms"]["total"][50]))
    print(f"best prec@1   : {best_p1['strategy']} ({best_p1['precision_at_1']:.3f})")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.pop("_raw", None)  # keep the report readable
    out = REPORT_DIR / f"retrieval-{model_slug(args.model)}-{args.kind}.json"
    out.write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model,
                    "repeats": args.repeats,
                    "k": args.k,
                    "kind": args.kind,
                },
                "n_queries": len(queries),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
