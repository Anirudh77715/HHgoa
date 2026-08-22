"""Retrieval with per-stage timing.

One non-obvious correctness issue this handles: with parent_child chunking,
several top-k CHILDREN frequently belong to the same parent document. Returning
them as-is looks like five results but is really one context repeated five
times -- it wastes the generation prompt and, worse, makes a thin retrieval
look well-supported.

So we over-fetch children, then collapse to unique parents keeping each
parent's best-scoring child. `k` therefore means "k distinct source documents",
which is what a caller actually wants.

The second is that a bi-encoder cannot tell `begin` from `end`. In hybrid mode
BM25 is fused into the ranking to fix that -- see app/index/lexical.py for the
measurement. Fusion decides ORDER ONLY: the score attached to each hit stays the
dense cosine, because that is the number bench/calibrate.py calibrated the score
gate against. RRF outputs live on a reciprocal-rank scale where the top result
is always roughly the same value regardless of how good it is, so gating on a
fused score would silently disable the gate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings, settings as default_settings
from app.index.embed import Embedder
from app.index.lexical import BM25, tokenize
from app.index.store import VectorStore

# How many children to pull before collapsing to unique parents. 4x covers the
# observed worst case (one long terms clause producing many children) without
# scanning a meaningful fraction of a 92-vector index.
OVERFETCH = 4


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    kind: str
    score: float
    context_text: str
    embed_text: str
    meta: dict = field(default_factory=dict)

    def cite(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "url": self.url,
            "score": round(self.score, 4),
        }


@dataclass
class RetrievalResult:
    query: str
    hits: list[Retrieved]
    timings_ms: dict[str, float]
    # Highest dense cosine over the whole candidate set -- NOT necessarily
    # hits[0].score, because fusion can promote a chunk whose cosine is lower.
    # The score gate must use THIS: bench/calibrate.py swept its threshold
    # against the best available cosine, and gating on the reordered first hit
    # would silently apply a stricter cutoff than the one that was calibrated.
    gate_score: float = 0.0
    # Query terms that appear anywhere in the corpus vocabulary, and the total
    # distinct terms asked. term_hits == 0 means the query shares no words with
    # the source at all. See guardrails.check_vocabulary.
    term_hits: int = 0
    query_terms: int = 0

    @property
    def top_score(self) -> float:
        """Cosine of the top-ranked result, for display. Use gate_score to gate."""
        return self.hits[0].score if self.hits else 0.0

    def contexts(self) -> list[str]:
        return [h.context_text for h in self.hits]


class Retriever:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        t0 = time.perf_counter()
        # Embedder.__init__ warms the ONNX graph, so the first real query does
        # not pay graph optimization and skew p100.
        self.embedder = Embedder(model_name=self.cfg.embed_model)
        self.store = VectorStore.load(self.cfg.index_dir)

        self.hybrid = self.cfg.retrieval_mode == "hybrid"
        # Built from the stored payloads, so this needs no extra build step and
        # no extra artifact on disk -- the lexical index is a derived view of
        # the same chunks the vectors came from. Built in BOTH modes, because
        # the vocabulary guardrail depends on it even when fusion is off.
        self.bm25 = BM25([p["embed_text"] for p in self.store.payloads])
        if self.hybrid:
            self.matrix = self.store.vectors
        self.load_ms = (time.perf_counter() - t0) * 1000

    def term_overlap(self, query: str) -> tuple[int, int]:
        """(query terms found anywhere in the corpus, total distinct terms).

        Zero overlap means the query shares no vocabulary at all with the
        source material -- nonsense, or another language. See
        guardrails.check_vocabulary for why this catches what cosine cannot.
        """
        terms = set(tokenize(query))
        return sum(1 for t in terms if t in self.bm25.postings), len(terms)

    def _rrf_order(self, qvec: np.ndarray, query: str) -> tuple[list[int], np.ndarray]:
        """Fused chunk ordering plus the dense cosine for every chunk.

        Both rankings must cover the SAME candidate set for fusion to mean
        anything, so this scores all chunks rather than a dense top-k. At 91
        vectors that is the identical scan IndexFlatIP does internally.
        """
        dense = self.matrix @ qvec
        lex = self.bm25.scores(query)

        def ranks(scores: np.ndarray) -> np.ndarray:
            order = np.argsort(-scores)
            r = np.empty(len(scores), dtype=np.int32)
            r[order] = np.arange(1, len(scores) + 1)
            return r

        kk = self.cfg.rrf_k
        fused = 1.0 / (kk + ranks(dense)) + 1.0 / (kk + ranks(lex))
        return list(np.argsort(-fused)), dense

    def search(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.cfg.top_k

        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        t1 = time.perf_counter()

        if self.hybrid:
            order, dense = self._rrf_order(qvec, query)
            # Scores stay dense cosines: fusion reorders, it does not rescore.
            raw = [(float(dense[i]), self.store.payloads[i]) for i in order[: k * OVERFETCH]]
            gate_score = float(dense.max()) if len(dense) else 0.0
        else:
            raw = self.store.search(qvec, k=k * OVERFETCH)
            gate_score = raw[0][0] if raw else 0.0
        t2 = time.perf_counter()

        # Collapse to unique parents, best child wins. Results arrive ordered
        # best-first, so the first occurrence of a doc_id is its best.
        seen: set[str] = set()
        hits: list[Retrieved] = []
        for score, p in raw:
            if p["doc_id"] in seen:
                continue
            seen.add(p["doc_id"])
            hits.append(
                Retrieved(
                    chunk_id=p["chunk_id"],
                    doc_id=p["doc_id"],
                    title=p["title"],
                    url=p["url"],
                    kind=p["kind"],
                    score=score,
                    context_text=p["context_text"],
                    embed_text=p["embed_text"],
                    meta=p.get("meta", {}),
                )
            )
            if len(hits) >= k:
                break
        t3 = time.perf_counter()

        term_hits, query_terms = self.term_overlap(query)
        return RetrievalResult(
            query=query,
            hits=hits,
            gate_score=gate_score,
            term_hits=term_hits,
            query_terms=query_terms,
            timings_ms={
                "embed": round((t1 - t0) * 1000, 3),
                "search": round((t2 - t1) * 1000, 3),
                "dedupe": round((t3 - t2) * 1000, 3),
                "total": round((t3 - t0) * 1000, 3),
            },
        )
