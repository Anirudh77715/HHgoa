"""Corpus invariants, enforced in CI.

Every check here exists because violating it produces a *confidently wrong
answer* rather than a visible crash — the worst failure mode for a RAG system
and the one a demo audience notices immediately.

Checks are structural, not content-exact: the site is allowed to reword copy
without breaking CI, but it is not allowed to silently lose the date->event
mapping or reintroduce the animated-counter zeros.

Run:  py -m app.corpus.verify
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

DOCS = Path("data/corpus/docs.json")

MIN_DOCS = 25
MIN_TIMELINE = 8
MIN_FAQ = 5
MIN_CHARS = 7000


def main() -> int:
    if not DOCS.exists():
        print(f"FAIL: {DOCS} missing — run `py -m app.corpus.scrape` first")
        return 1

    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    failures: list[str] = []
    warnings: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            failures.append(msg)
        print(f"  {'ok  ' if ok else 'FAIL'}  {msg}")

    print("corpus invariants:")

    # --- size ---
    check(len(docs) >= MIN_DOCS, f"at least {MIN_DOCS} docs (got {len(docs)})")
    total = sum(len(d["text"]) for d in docs)
    check(total >= MIN_CHARS, f"at least {MIN_CHARS} chars (got {total})")

    # --- the animated-counter trap ---
    # These render as 0 in server HTML. If they ever reappear in the corpus,
    # the system will state "0 hackers attended" with full confidence.
    leaked = [d["doc_id"] for d in docs if re.search(r"\$?0k\+|\b0\+", d["text"])]
    check(not leaked, f"no animated-counter zeros leaked (offenders: {leaked or 'none'})")

    # --- timeline date pairing ---
    # A positional CSS grid pairs dates to events by index only. If the markup
    # changes, get_text() silently flattens it and every date answer goes wrong.
    tl = [d for d in docs if d["kind"] == "timeline"]
    check(len(tl) >= MIN_TIMELINE, f"at least {MIN_TIMELINE} timeline milestones (got {len(tl)})")
    undated = [d["doc_id"] for d in tl if not d.get("meta", {}).get("date")]
    check(not undated, f"every milestone carries a date (missing: {undated or 'none'})")
    fellback = [d["doc_id"] for d in tl if d.get("meta", {}).get("extraction")]
    check(not fellback, f"timeline used DOM pairing, not a fallback ({fellback or 'none'})")

    # --- faq pairing ---
    faq = [d for d in docs if d["kind"] == "faq"]
    check(len(faq) >= MIN_FAQ, f"at least {MIN_FAQ} FAQs (got {len(faq)})")
    unpaired = [d["doc_id"] for d in faq if "Q:" not in d["text"] or "A:" not in d["text"]]
    check(not unpaired, f"every FAQ keeps question with answer (bad: {unpaired or 'none'})")

    # --- general hygiene ---
    untitled = [d["doc_id"] for d in docs if not d["title"].strip()]
    check(not untitled, f"no untitled docs ({untitled or 'none'})")

    dupes = [t for t, c in Counter(d["text"] for d in docs).items() if c > 1]
    check(not dupes, f"no duplicate bodies ({len(dupes)} found)")

    ids = [d["doc_id"] for d in docs]
    check(len(ids) == len(set(ids)), "doc_ids are unique")

    # Volatile content must never be indexed by default: /result is a live
    # leaderboard, so any answer drawn from it is stale on the next reveal.
    vol = [d["doc_id"] for d in docs if d.get("volatile")]
    check(not vol, f"no volatile docs indexed ({vol or 'none'})")

    # --- warnings (not failures) ---
    for d in docs:
        if d.get("meta", {}).get("extraction"):
            warnings.append(f"{d['doc_id']} used fallback extraction: {d['meta']['extraction']}")
    tiny = [d["doc_id"] for d in docs if len(d["text"]) < 40]
    if tiny:
        warnings.append(f"very short docs (weak retrieval targets): {tiny}")

    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  warn  {w}")

    print()
    if failures:
        print(f"{len(failures)} invariant(s) FAILED")
        return 1
    print(f"all invariants passed — {len(docs)} docs, {total} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
