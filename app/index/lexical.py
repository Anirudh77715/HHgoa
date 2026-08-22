"""BM25 lexical scoring, to sit alongside the dense index.

WHY THIS EXISTS

The dense retriever ranked `timeline-08` *Registration Ends — 1 October 2026*
ABOVE `timeline-00` *Registration Begins — 7 May 2026* for the query "when does
registration begin". The two passages are near-identical apart from one antonym,
and a bi-encoder cosine does not encode polarity: `begin` and `end` occupy
almost the same region of embedding space because they are the same *kind* of
word about the same *kind* of event.

Lexical scoring does not have that problem. "begin" is simply a different term
from "end", so BM25 separates the two exactly where the embedding cannot.
MEASURED on the 48-query eval set (see bench/bench_hybrid.py): BM25 alone puts
the correct passage at rank 1 for that query, and every dense+lexical fusion
tested inherits the fix.

WHY NO DEPENDENCY

`rank_bm25` would do, but this is ~60 lines against a 91-chunk index and the
project already pays deliberate attention to image size (see the fastembed
rationale in requirements.txt). Writing it also means controlling tokenisation,
which is the part that actually decides whether the antonym case works.

WHY THE CRUDE STEMMER

The query says "begin"; the document says "Begins". With no suffix handling BM25
contributes nothing at all to the case it was added for. A real stemmer means an
nltk dependency and a data download; this strips the three suffixes that matter
on this corpus and leaves everything else alone. It is not linguistically
correct -- "gas" -> "ga" -- but it is symmetric (applied to query and document
alike), so a wrong stem still matches itself. Documented rather than hidden.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

WORD_RE = re.compile(r"[a-z0-9]+")

# Order matters: check longer suffixes first so "endings" -> "end", not "ending".
_SUFFIXES = ("ing", "es", "s")


def stem(word: str) -> str:
    for suf in _SUFFIXES:
        # Keep a 2-char minimum stem so short words are left intact.
        if len(word) > len(suf) + 2 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in WORD_RE.findall(text.lower())]


class BM25:
    """Okapi BM25 over an in-memory postings list.

    Scoring iterates query terms and their postings rather than looping every
    document, so cost scales with how many documents actually contain a query
    term -- typically a handful here, not the whole index.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(texts)

        docs = [tokenize(t) for t in texts]
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        # An empty index would make avgdl a division by zero; there is nothing
        # sensible to score against, so guard rather than emit NaNs.
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 1.0

        postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                postings.setdefault(term, []).append((i, tf))

        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for term, plist in postings.items():
            idx = np.array([p[0] for p in plist], dtype=np.int32)
            tfs = np.array([p[1] for p in plist], dtype=np.float32)
            self.postings[term] = (idx, tfs)
            df = len(plist)
            # BM25+ style idf: always positive, so a term appearing in most
            # documents contributes little instead of scoring negatively.
            self.idf[term] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def __len__(self) -> int:
        return self.n_docs

    def scores(self, query: str) -> np.ndarray:
        """BM25 score per document, aligned with the order passed to __init__."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return out

        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in tokenize(query):
            hit = self.postings.get(term)
            if hit is None:
                continue
            idx, tfs = hit
            out[idx] += self.idf[term] * tfs * (self.k1 + 1) / (tfs + norm[idx])
        return out
