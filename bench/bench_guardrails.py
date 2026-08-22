"""Acceptance test for the guardrail chain, groundedness in particular.

CONTEXT.md recorded groundedness as "untested, blocked by the 401". That was a
misreading of the dependency. `check_groundedness(answer, contexts)` is a pure
function of two strings -- it never touches the network. Only the end-to-end
path needs a Gemini key. So the guardrail that the whole design leans on can be
held to account today, and its determinism is exactly what makes this a test
rather than a sample of one model's mood.

What this measures, per case in bench/guardrail_cases.yaml:

  gate chain    which of scope -> score -> groundedness fires first. A case
                "caught" by the score gate tells you nothing about
                groundedness, so the two are reported separately rather than
                blurred into one pass rate.

  verdict       whether groundedness itself did the right thing, given real
                contexts retrieved live from the current index.

Known limits are declared in the case file and reported apart from the score.
A guardrail evaluated only on cases it was built to pass is marketing. The
number that matters is the undocumented failures, and the run exits non-zero
only on those -- a documented blind spot is not a regression, but a NEW one is.

Run:  py -m bench.bench_guardrails
      py -m bench.bench_guardrails --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from app.config import settings
from app.rag import guardrails as g
from app.rag.retrieve import Retriever

CASES = Path("bench/guardrail_cases.yaml")
REPORT_DIR = Path("bench/reports")


def load_cases() -> list[dict]:
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    for c in cases:
        c.setdefault("known_limit", False)
        c.setdefault("answer", "")
        if c.get("gate_expect"):
            if c["gate_expect"] not in ("block", "pass"):
                raise ValueError(f"{c['id']}: gate_expect must be block|pass")
        elif c.get("expect") not in ("block", "pass"):
            raise ValueError(f"{c['id']}: expect must be block|pass")
    return cases


def first_gate(query: str, res, threshold: float) -> str:
    """Which gate stops this query before generation is even attempted."""
    if not g.check_scope(query).ok:
        return "scope"
    if not g.check_vocabulary(res.term_hits, res.query_terms).ok:
        return "vocabulary"
    if not g.check_score(res.gate_score, threshold).ok:
        return "score"
    return "reaches_generation"


def main() -> int:
    ap = argparse.ArgumentParser(description="Acceptance test for the guardrail chain.")
    ap.add_argument("--verbose", action="store_true", help="show unsupported tokens")
    args = ap.parse_args()

    cases = load_cases()
    retriever = Retriever()
    threshold = settings.retrieval_min_score

    print(f"cases     : {len(cases)}")
    print(f"index     : {len(retriever.store)} vectors, strategy={settings.strategy}")
    print(f"threshold : {threshold}\n")

    rows: list[dict] = []
    # Retrieval is deterministic, so cache it: several cases share a query and
    # re-embedding would only add noise to the run time.
    ctx_cache: dict[str, tuple] = {}

    for c in cases:
        q = c["query"]
        if q not in ctx_cache:
            r = retriever.search(q)
            ctx_cache[q] = (r.contexts(), r.gate_score, r)
        contexts, top_score, res = ctx_cache[q]

        gate = first_gate(q, res, threshold)
        if c.get("gate_expect"):
            # Pre-generation case: what matters is whether a gate stopped it,
            # not what groundedness would have said about an answer we never
            # generate. Reported in the same table so one run covers the chain.
            v = g.Verdict(gate == "reaches_generation", "stopped_at_" + gate, {})
            acted = "pass" if gate == "reaches_generation" else "block"
            correct = acted == c["gate_expect"]
        else:
            v = g.check_groundedness(c["answer"], contexts)
            acted = "pass" if v.ok else "block"
            correct = acted == c["expect"]

        rows.append(
            {
                "id": c["id"],
                "query": q,
                "expect": c.get("gate_expect") or c["expect"],
                "acted": acted,
                "correct": correct,
                "known_limit": bool(c["known_limit"]),
                "reason": v.reason,
                "first_gate": gate,
                "top_score": round(top_score, 4),
                "unsupported": v.detail.get("unsupported", []),
                "checked": v.detail.get("checked", 0),
            }
        )

    # --- results ------------------------------------------------------------
    print(f"{'id':<7}{'expect':<8}{'acted':<8}{'':<3}{'first gate':<20}{'reason':<24}checked")
    for r in rows:
        if r["correct"]:
            mark = "ok "
        elif r["known_limit"]:
            mark = "LIM"
        else:
            mark = "!! "
        print(
            f"{r['id']:<7}{r['expect']:<8}{r['acted']:<8}{mark:<5}"
            f"{r['first_gate']:<20}{r['reason']:<24}{r['checked']}"
        )
        if args.verbose and r["unsupported"]:
            print(f"       unsupported: {r['unsupported']}")

    scored = [r for r in rows if not r["known_limit"]]
    limits = [r for r in rows if r["known_limit"]]
    passed = [r for r in scored if r["correct"]]
    regressions = [r for r in scored if not r["correct"]]
    # A known limit that starts behaving correctly is good news, but it means
    # the case file is stale and should be reclassified rather than left lying.
    fixed = [r for r in limits if r["correct"]]

    blocked_earlier = [r for r in rows if r["first_gate"] != "reaches_generation"]

    print()
    print(f"scored cases      : {len(passed)}/{len(scored)} correct")
    print(f"documented limits : {len(limits)} (excluded from the score, listed below)")
    if fixed:
        print(f"limits now PASSING: {[r['id'] for r in fixed]} — reclassify these in the case file")
    print(f"never reach generation (stopped by scope/score): {len(blocked_earlier)}/{len(rows)}")

    print("\ndocumented blind spots — groundedness does NOT catch these:")
    for r in limits:
        if not r["correct"]:
            print(f"  {r['id']}  [{r['reason']}]  {r['query']}")

    if regressions:
        print(f"\n{len(regressions)} UNDOCUMENTED failure(s):")
        for r in regressions:
            print(f"  {r['id']}  expected {r['expect']}, got {r['acted']} ({r['reason']})")
            if r["unsupported"]:
                print(f"        unsupported: {r['unsupported']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "guardrails.json"
    out.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "strategy": settings.strategy,
                "n_cases": len(rows),
                "scored_correct": len(passed),
                "scored_total": len(scored),
                "documented_limits": len(limits),
                "undocumented_failures": [r["id"] for r in regressions],
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {out}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
