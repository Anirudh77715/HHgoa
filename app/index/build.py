"""Build one FAISS index per chunking strategy.

Separate indexes (rather than one tagged index) so the benchmark can measure
each strategy in isolation. Sharing an index would let one strategy's chunks
crowd another's out of top-k and make the comparison meaningless.

Run:  py -m app.index.build            # flat, all strategies
      py -m app.index.build --kind hnsw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.index.embed import DEFAULT_MODEL, Embedder, model_slug
from app.index.store import VectorStore

CHUNK_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")
STRATEGIES = ("structural", "recursive", "parent_child")


def index_path(model: str, kind: str, strategy: str) -> Path:
    return INDEX_DIR / model_slug(model) / kind / strategy


def build_one(name: str, embedder: Embedder, kind: str) -> dict:
    src = CHUNK_DIR / f"{name}.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing — run `py -m app.chunking.build` first")

    chunks = json.loads(src.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    vectors = embedder.embed_passages([c["embed_text"] for c in chunks])
    embed_ms = (time.perf_counter() - t0) * 1000

    store = VectorStore(dim=embedder.dim, kind=kind)
    # Payload carries context_text, not embed_text: retrieval matches on the
    # embedded text but the model must answer from the context.
    store.add(
        vectors,
        [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "strategy": c["strategy"],
                "title": c["title"],
                "url": c["url"],
                "kind": c["kind"],
                "context_text": c["context_text"],
                "embed_text": c["embed_text"],
                "meta": c.get("meta", {}),
            }
            for c in chunks
        ],
    )

    out = index_path(embedder.model_name, kind, name)
    store.save(out)

    return {
        "strategy": name,
        "kind": kind,
        "model": embedder.model_name,
        "n_vectors": len(store),
        "dim": embedder.dim,
        "embed_total_ms": round(embed_ms, 1),
        "embed_per_chunk_ms": round(embed_ms / max(len(chunks), 1), 2),
        "path": str(out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FAISS indexes per chunking strategy.")
    ap.add_argument("--kind", choices=("flat", "hnsw"), default="flat")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    print("loading embedder (includes ONNX warmup)...")
    t0 = time.perf_counter()
    embedder = Embedder(model_name=args.model)
    print(f"  model={embedder.model_name} dim={embedder.dim} "
          f"load+warmup={((time.perf_counter() - t0) * 1000):.0f}ms\n")

    results = []
    for name in STRATEGIES:
        r = build_one(name, embedder, args.kind)
        results.append(r)
        print(f"=== {name} ({args.kind}) ===")
        print(f"  vectors : {r['n_vectors']}  dim {r['dim']}")
        print(f"  embed   : {r['embed_total_ms']}ms total, "
              f"{r['embed_per_chunk_ms']}ms/chunk")
        print(f"  saved   : {r['path']}\n")

    out_root = INDEX_DIR / model_slug(embedder.model_name)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"build-{args.kind}.json").write_text(
        json.dumps({"model": embedder.model_name, "indexes": results}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
