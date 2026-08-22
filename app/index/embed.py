"""Local embeddings via fastembed (ONNX Runtime).

Two decisions here carry the latency budget:

1.  ONNX over PyTorch. Same BAAI/bge-small-en-v1.5 weights, but no torch in
    the image and measurably faster CPU inference. On a free 2-vCPU host that
    difference is the whole sub-50ms retrieval claim.

2.  Asymmetric encoding. bge-* models are trained with an instruction prefix
    on the QUERY side only ("Represent this sentence for searching relevant
    passages:"). Embedding queries and passages identically is the single most
    common way to silently lose retrieval quality with these models, so we
    route queries through query_embed() and passages through passage_embed().

Warmup is explicit and non-optional: the first ONNX call pays graph
optimization and arena allocation, which is easily 10-50x a steady-state call.
Leaving that in the measurement would make p100 meaningless.
"""

from __future__ import annotations

import os

import numpy as np
from fastembed import TextEmbedding

# NOTE ON THE DEFAULT -- measured, not assumed.
#
# fastembed ships BAAI/bge-small-en-v1.5 as an int8-QUANTIZED ONNX build
# ("-onnx-q"). On this hardware (Ryzen 7 5800H, 249 GFLOPS on fp32 matmul) it
# takes ~153ms p50 to embed a 14-token query, while the NON-quantized
# BAAI/bge-small-en takes ~6.7ms -- the quantized model is 23x SLOWER. The
# cause is unfused quantize/dequantize nodes: every op pays a quant->compute
# ->dequant round trip that dwarfs the actual matmul at this input size.
#
# That single default would have eaten the entire latency budget while looking
# like the obvious "small and optimized" choice. Model identity is therefore a
# benchmark axis (see bench/bench_models.py), not a hardcoded guess.
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")


def model_slug(name: str) -> str:
    """Filesystem-safe id, so indexes for different models never collide."""
    return name.replace("/", "__")


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize so inner product == cosine similarity.

    Normalizing here means FAISS can use a plain inner-product index and the
    scores are directly comparable to the RETRIEVAL_MIN_SCORE guardrail
    threshold, which only makes sense on a bounded [-1, 1] scale.
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir or os.getenv("FASTEMBED_CACHE_PATH"),
        )
        self.dim = int(self.warmup().shape[-1])

    def warmup(self) -> np.ndarray:
        """Force graph optimization now, outside any measured path."""
        return self.embed_query("warmup")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.array(list(self._model.passage_embed(texts)), dtype=np.float32)
        return _unit(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.query_embed([text])))
        return _unit(np.asarray(vec, dtype=np.float32))
