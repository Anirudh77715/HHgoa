"""Build all chunk sets and report comparable stats.

Emits one file per strategy so retrieval can be benchmarked per strategy and
the winner argued from numbers. Deduplicates within a strategy: overlap and
title-prefixing can produce identical embed_text, and duplicate vectors waste
index space while skewing top-k.

Run:  py -m app.chunking.build
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from app.chunking.strategies import STRATEGIES, Chunk

DOCS = Path("data/corpus/docs.json")
OUT_DIR = Path("data/chunks")


def dedupe(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        key = c.embed_text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out, len(chunks) - len(out)


def stats(chunks: list[Chunk]) -> dict:
    embed = [c.n_embed for c in chunks]
    ctx = [c.n_context for c in chunks]
    return {
        "n_chunks": len(chunks),
        "embed_chars": {
            "min": min(embed),
            "median": int(statistics.median(embed)),
            "mean": round(statistics.mean(embed), 1),
            "max": max(embed),
        },
        "context_chars": {
            "min": min(ctx),
            "median": int(statistics.median(ctx)),
            "mean": round(statistics.mean(ctx), 1),
            "max": max(ctx),
        },
        "amplification": round(sum(ctx) / sum(embed), 2),
    }


def main() -> int:
    if not DOCS.exists():
        print(f"FAIL: {DOCS} missing — run `py -m app.corpus.scrape` first")
        return 1

    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    for name, fn in STRATEGIES.items():
        chunks, dropped = dedupe(fn(docs))
        s = stats(chunks)
        s["deduped"] = dropped
        summary[name] = s

        (OUT_DIR / f"{name}.json").write_text(
            json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"=== {name} ===")
        print(f"  chunks       : {s['n_chunks']}  (deduped {dropped})")
        e, c = s["embed_chars"], s["context_chars"]
        print(f"  embed  chars : min {e['min']:>4}  med {e['median']:>4}  max {e['max']:>4}")
        print(f"  ctx    chars : min {c['min']:>4}  med {c['median']:>4}  max {c['max']:>4}")
        print(f"  amplification: {s['amplification']}x  (context read / text embedded)")
        print()

    (OUT_DIR / "summary.json").write_text(
        json.dumps({"n_docs": len(docs), "strategies": summary}, indent=2), encoding="utf-8"
    )
    print(f"written: {OUT_DIR}/[{', '.join(STRATEGIES)}].json + summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
