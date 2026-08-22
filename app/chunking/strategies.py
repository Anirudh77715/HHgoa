"""Three chunking strategies, run side by side so the choice is measured
rather than asserted.

The central abstraction is that every chunk carries TWO texts:

    embed_text    what we embed and search over  (precision)
    context_text  what we hand the LLM to answer from  (recall)

Most naive RAG sets these equal, which forces one tradeoff for both jobs: big
chunks retrieve imprecisely, small chunks retrieve well but hand the model a
fragment with the answer cut off. Splitting them lets each strategy tune the
two independently, and it is what makes parent-child work at all.

Strategies
----------
structural    One chunk per source doc. The corpus builder already emits
              semantic units (one FAQ, one milestone, one terms clause), so
              this is the "respect the document's own structure" baseline.
              embed == context.

recursive     Separator-hierarchy splitting with overlap. The generic
              fallback: no structural knowledge, just paragraph -> line ->
              sentence -> word. Included as the honest baseline to beat,
              since it is what most RAG tutorials ship.

parent_child  Embed sentence-sized children for retrieval precision, but
              return the whole parent doc as context. Fixes the dominant
              failure of small-chunk retrieval: matching the right fragment
              and then answering from too little surrounding text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]

RECURSIVE_SIZE = 350
RECURSIVE_OVERLAP = 80
CHILD_SIZE = 160


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    embed_text: str
    context_text: str
    title: str
    url: str
    kind: str
    meta: dict = field(default_factory=dict)

    @property
    def n_embed(self) -> int:
        return len(self.embed_text)

    @property
    def n_context(self) -> int:
        return len(self.context_text)


# --------------------------------------------------------------------------
# splitting primitives
# --------------------------------------------------------------------------


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    """Split `text` to pieces <= size, descending the separator hierarchy.

    Descends only for the parts that are still too long, so a document with one
    runaway paragraph does not get shredded at word level everywhere else.
    """
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    if not seps:
        # No separator left: hard-cut. Reached only by pathological input
        # (a single unbroken 350+ char token).
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = seps[0], seps[1:]
    if sep not in text:
        return _split_recursive(text, size, rest)

    out: list[str] = []
    cur = ""
    for part in text.split(sep):
        candidate = f"{cur}{sep}{part}" if cur else part
        if len(candidate) <= size:
            cur = candidate
            continue
        if cur:
            out.append(cur)
        if len(part) > size:
            out.extend(_split_recursive(part, size, rest))
            cur = ""
        else:
            cur = part
    if cur:
        out.append(cur)
    return [c.strip() for c in out if c.strip()]


def _with_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prefix each piece with the tail of its predecessor.

    Overlap exists so a fact straddling a boundary survives in at least one
    chunk. Applied after splitting rather than during, so the split points stay
    at real separators.
    """
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        # Start the carried tail at a word boundary so we do not paste a
        # half-word onto the front of the next chunk.
        if " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        out.append(f"{tail} {cur}".strip())
    return out


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _pack(units: list[str], size: int) -> list[str]:
    """Greedily pack units into groups of at most `size` chars."""
    out: list[str] = []
    cur = ""
    for u in units:
        candidate = f"{cur} {u}".strip()
        if cur and len(candidate) > size:
            out.append(cur)
            cur = u
        else:
            cur = candidate
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------
# strategies
# --------------------------------------------------------------------------


def _header(doc: dict) -> str:
    """Prefix chunks with their document title.

    A bare terms clause or timeline description is close to unembeddable on its
    own — "Final shortlist confirmed before partner matching." carries almost
    no retrievable signal without "Delta Selections" attached.
    """
    title = doc["title"].strip()
    section = doc.get("section", "").strip()
    if section and section.lower() not in title.lower():
        return f"{section} — {title}"
    return title


def structural_chunks(docs: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.append(
            Chunk(
                chunk_id=f"st::{doc['doc_id']}",
                doc_id=doc["doc_id"],
                strategy="structural",
                embed_text=doc["text"],
                context_text=doc["text"],
                title=doc["title"],
                url=doc["url"],
                kind=doc["kind"],
                meta={**doc.get("meta", {}), "n_parts": 1},
            )
        )
    return out


def recursive_chunks(
    docs: list[dict],
    size: int = RECURSIVE_SIZE,
    overlap: int = RECURSIVE_OVERLAP,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        pieces = _with_overlap(_split_recursive(doc["text"], size, SEPARATORS), overlap)
        for i, piece in enumerate(pieces):
            out.append(
                Chunk(
                    chunk_id=f"rc::{doc['doc_id']}::{i:02d}",
                    doc_id=doc["doc_id"],
                    strategy="recursive",
                    embed_text=piece,
                    context_text=piece,
                    title=doc["title"],
                    url=doc["url"],
                    kind=doc["kind"],
                    meta={**doc.get("meta", {}), "part": i, "n_parts": len(pieces)},
                )
            )
    return out


def parent_child_chunks(docs: list[dict], child_size: int = CHILD_SIZE) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        parent = doc["text"]
        children = _pack(_sentences(parent), child_size)
        if not children:
            continue
        head = _header(doc)
        for i, child in enumerate(children):
            # Title-prefix the CHILD (what we embed) but leave the parent
            # (what the model reads) untouched.
            embed_text = child if head.lower() in child.lower() else f"{head}: {child}"
            out.append(
                Chunk(
                    chunk_id=f"pc::{doc['doc_id']}::{i:02d}",
                    doc_id=doc["doc_id"],
                    strategy="parent_child",
                    embed_text=embed_text,
                    context_text=parent,
                    title=doc["title"],
                    url=doc["url"],
                    kind=doc["kind"],
                    meta={**doc.get("meta", {}), "child": i, "n_children": len(children)},
                )
            )
    return out


STRATEGIES = {
    "structural": structural_chunks,
    "recursive": recursive_chunks,
    "parent_child": parent_child_chunks,
}
