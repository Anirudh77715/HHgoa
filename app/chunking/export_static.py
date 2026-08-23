"""Emit the corpus as a static JSON the browser can retrieve over, unaided.

WHY THIS EXISTS

The API cannot run on a static host. Netlify Functions are JS/Go only, and the
Python service needs ONNX Runtime, FAISS and a ~130MB model. So a frontend-only
deploy has no backend, and every question fails.

This exports the chunks so the page can do retrieval ITSELF as a fallback --
"lite mode". Honest about what that costs: the browser has no embedding model,
so lite mode is **BM25 only**, no dense vectors, no fusion. Measured on the same
37-query eval set (bench/bench_hybrid.py):

    ranker        recall@5   prec@1   MRR
    hybrid (API)     0.919    0.703  0.763    <- the real pipeline
    bm25 only        0.865    0.649  0.733    <- what lite mode can do

So lite mode is a genuine downgrade, not a free lunch, and the UI says so. It
exists because a working degraded demo beats a dead page, and because it keeps
the guardrails and citations intact -- only ranking quality drops.

context_text is deduplicated by parent: with parent_child chunking many
children share one parent, and repeating it inline nearly triples the payload
(84KB -> 37KB).

Run:  py -m app.chunking.export_static
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CHUNKS = Path("data/chunks/parent_child.json")
OUT = Path("app/static/corpus.json")


def main() -> int:
    if not CHUNKS.exists():
        print(f"FAIL: {CHUNKS} missing — run `py -m app.chunking.build` first")
        return 1

    raw = json.loads(CHUNKS.read_text(encoding="utf-8"))
    chunks = raw["chunks"] if isinstance(raw, dict) else raw

    parents: dict[str, str] = {}
    out_chunks = []
    for c in chunks:
        parents.setdefault(c["doc_id"], c["context_text"])
        out_chunks.append(
            {
                "c": c["chunk_id"],
                "d": c["doc_id"],
                "t": c["title"],
                "u": c["url"],
                "k": c.get("kind", ""),
                "e": c["embed_text"],
            }
        )

    payload = {
        "note": (
            "Lite-mode corpus for browser-only retrieval. BM25 only — no dense "
            "vectors, because the browser has no embedding model. Measured "
            "recall@5 0.865 / prec@1 0.649 against the API's hybrid 0.919 / 0.703."
        ),
        "n_chunks": len(out_chunks),
        "n_docs": len(parents),
        "chunks": out_chunks,
        "parents": parents,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  —  {len(out_chunks)} chunks, {len(parents)} docs, {kb:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
