"""FAISS vector store.

On index choice, stated plainly because it would be easy to cargo-cult here:

    This corpus is 33-92 vectors. An exact flat inner-product scan over 92
    vectors of dim 384 is ~35k multiply-adds -- microseconds, and EXACT. HNSW
    is an approximate index that trades recall for sublinear search; at this
    scale it is strictly worse on both axes (it adds graph-traversal overhead
    AND can miss the true nearest neighbour).

So `flat` is the default and the honest choice. `hnsw` is implemented anyway so
the benchmark can show the crossover empirically rather than us asserting it --
a reviewer can run both and see that flat wins until the corpus is orders of
magnitude larger.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


class VectorStore:
    def __init__(self, dim: int, kind: str = "flat") -> None:
        if kind not in ("flat", "hnsw"):
            raise ValueError(f"unknown index kind: {kind}")
        self.dim = dim
        self.kind = kind
        self.payloads: list[dict[str, Any]] = []

        if kind == "flat":
            # Exact inner product. Vectors are unit-normalized upstream, so
            # inner product is cosine similarity.
            self.index: faiss.Index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            self.index.hnsw.efSearch = HNSW_EF_SEARCH

    def __len__(self) -> int:
        return int(self.index.ntotal)

    @property
    def vectors(self) -> np.ndarray:
        """All stored vectors as an (n, dim) matrix.

        Hybrid retrieval needs a dense score for EVERY chunk, not just top-k,
        because fusion has to rank the same candidate set that BM25 scored. At
        this corpus size reconstructing is trivial, and `matrix @ q` is the
        same exact inner-product scan IndexFlatIP performs internally.
        """
        if len(self) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.index.reconstruct_n(0, len(self))

    def add(self, vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        if len(vectors) != len(payloads):
            raise ValueError(f"vectors ({len(vectors)}) != payloads ({len(payloads)})")
        if len(vectors) == 0:
            return
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.payloads.extend(payloads)

    def search(self, qvec: np.ndarray, k: int = 5) -> list[tuple[float, dict[str, Any]]]:
        if len(self) == 0:
            return []
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)
        k = min(k, len(self))
        scores, idxs = self.index.search(q, k)
        out: list[tuple[float, dict[str, Any]]] = []
        for score, i in zip(scores[0], idxs[0]):
            # FAISS returns -1 for empty slots when k > ntotal.
            if i < 0:
                continue
            out.append((float(score), self.payloads[int(i)]))
        return out

    # --- persistence -----------------------------------------------------

    def save(self, dirpath: str | Path) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        with (d / "payloads.pkl").open("wb") as f:
            pickle.dump(self.payloads, f)
        (d / "meta.json").write_text(
            json.dumps({"dim": self.dim, "kind": self.kind, "n": len(self)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dirpath: str | Path) -> VectorStore:
        d = Path(dirpath)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"], kind=meta["kind"])
        store.index = faiss.read_index(str(d / "index.faiss"))
        with (d / "payloads.pkl").open("rb") as f:
            store.payloads = pickle.load(f)
        return store
