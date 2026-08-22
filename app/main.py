"""FastAPI service: voice question -> grounded answer.

Pipeline, with the guardrail placement that the benchmark forced:

    query
      -> scope filter        (free, pre-retrieval; catches imperatives)
      -> retrieve            (~7ms, local)
      -> vocabulary gate     (free; rejects input sharing NO words with the
                              corpus -- nonsense and other languages, which
                              cosine cannot distinguish from real questions)
      -> score gate          (free; a COST filter, blocks 5/11 refusals)
      -> generate            (network; the only external call)
      -> groundedness check  (free; the gate that actually stops fabrication)
      -> answer | refusal

Generation is optional. Without a key -- or once the free-tier quota is spent --
the service answers in EXTRACTIVE mode, returning the best retrieved passage
verbatim. Same code path both times, so the degraded mode is continuously
exercised instead of being untested emergency code.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.rag import guardrails as g
from app.rag.retrieve import Retriever

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
log = logging.getLogger("hhgoa-rag")

STATIC_DIR = "app/static"

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and index at startup, never per request.

    A lazy first-request load would put a multi-second model load inside a
    measured request and make p100 meaningless.
    """
    t0 = time.perf_counter()
    state["retriever"] = Retriever(settings)
    log.info(
        "ready in %.0fms | %s | index=%s | generation=%s",
        (time.perf_counter() - t0) * 1000,
        settings.embed_model,
        settings.index_dir,
        "on" if settings.can_generate else "OFF (extractive mode)",
    )
    yield
    state.clear()


app = FastAPI(title="Ask HH Goa", version="0.1.0", lifespan=lifespan)

# The frontend can be served from this app OR hosted separately (Netlify, which
# cannot run this Python backend -- see DEPLOY.md). A split origin means the
# browser sends cross-origin requests, so CORS has to allow them.
#
# ALLOWED_ORIGINS is an explicit comma-separated list; "*" is accepted for a
# public read-only demo. Credentials are deliberately NOT allowed: the BYOK key
# travels in an explicit header, never a cookie, so there is nothing
# ambient for another origin to ride on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Gemini-Key"],
)


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    doc_id: str
    title: str
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    mode: Literal["generated", "extractive", "refused"]
    answered: bool
    citations: list[Citation]
    # Every refusal names the gate that fired, so a reviewer can tell a
    # deliberate refusal from a silent failure.
    refusal_reason: str | None = None
    guardrails: dict[str, Any]
    timings_ms: dict[str, float]


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    r = state.get("retriever")
    return {
        "status": "ok" if r else "loading",
        "vectors": len(r.store) if r else 0,
        "generation_enabled": settings.can_generate,
        # Tells the frontend whether to offer the BYOK prompt. The service is
        # fully usable either way; this only decides what the UI suggests.
        "accepts_user_key": True,
    }


@app.get("/config")
def config() -> dict:
    """Redacted config. Never exposes the API key."""
    return settings.redacted()


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    x_gemini_key: str | None = Header(default=None, alias="X-Gemini-Key"),
) -> AskResponse:
    """Bring-your-own-key is optional by design.

    With no key the service answers EXTRACTIVELY from the top passage -- the
    same degraded path it uses when quota runs out, so it is continuously
    exercised rather than being untested emergency code. A key upgrades the
    answer to generated prose; it never gates the pipeline.

    The key is used for this request and discarded: never logged, never stored,
    never echoed back in the response.
    """
    byok = (x_gemini_key or "").strip() or None
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    checks: dict[str, Any] = {}

    def refuse(reason: str, extra: dict | None = None) -> AskResponse:
        timings["total"] = round((time.perf_counter() - t_start) * 1000, 3)
        return AskResponse(
            # A vocabulary miss gets its own text: "I don't have that" would be
            # misleading for a question the corpus DOES cover, asked in a
            # language the English-only index cannot represent.
            answer=(extra or {}).get("text", g.REFUSAL_TEXT),
            mode="refused",
            answered=False,
            citations=(extra or {}).get("citations", []),
            refusal_reason=reason,
            guardrails=checks,
            timings_ms=timings,
        )

    # --- gate 1: scope (pre-retrieval, free) ---
    t0 = time.perf_counter()
    scope = g.check_scope(req.query)
    timings["scope"] = round((time.perf_counter() - t0) * 1000, 3)
    checks["scope"] = {"ok": scope.ok, "reason": scope.reason, **scope.detail}
    if not scope.ok:
        return refuse(scope.reason)

    # --- retrieve ---
    retriever: Retriever = state["retriever"]
    result = retriever.search(req.query, k=req.k)
    timings.update({f"retrieval_{k}": v for k, v in result.timings_ms.items()})

    citations = [Citation(**h.cite()) for h in result.hits]

    if not result.hits:
        return refuse("no_results", {"citations": citations})

    # --- gate 2: vocabulary (free, catches what cosine cannot) ---
    # A real voice test produced romanised Hindi, and "asdf qwer zxcv hjkl"
    # measured cosine 0.8317 -- above the gate and above several genuine
    # questions. An embedding always lands somewhere; zero lexical overlap is
    # the signal that separates strangers cleanly. See check_vocabulary.
    vocab = g.check_vocabulary(result.term_hits, result.query_terms)
    checks["vocabulary"] = {"ok": vocab.ok, "reason": vocab.reason, **vocab.detail}
    if not vocab.ok:
        return refuse(vocab.reason, {"citations": [], "text": g.NO_VOCAB_TEXT})

    # --- gate 3: score (cost filter, free) ---
    # gate_score, not top_score: under hybrid retrieval fusion can promote a
    # chunk with a lower cosine, and the threshold was calibrated against the
    # best available cosine. See RetrievalResult.gate_score.
    score = g.check_score(result.gate_score, settings.retrieval_min_score)
    checks["score"] = {"ok": score.ok, "reason": score.reason, **score.detail}
    if not score.ok:
        # Deliberately refuse without spending a generation call.
        return refuse(score.reason, {"citations": citations})

    # --- generate, or fall back to extractive ---
    contexts = result.contexts()
    if settings.can_generate or byok:
        from app.rag.generate import generate  # imported lazily: needs the SDK

        t0 = time.perf_counter()
        gen = generate(req.query, result.hits, api_key=byok)
        timings["generate"] = round((time.perf_counter() - t0) * 1000, 3)
        checks["generation"] = gen.diagnostics

        if not gen.ok:
            # Quota exhausted, timeout, or the model declined to answer.
            # Degrade rather than fail: the top passage is still useful.
            answer, mode = result.hits[0].context_text, "extractive"
            checks["degraded"] = {"reason": gen.reason}
        elif not gen.answerable:
            # The model itself reported it cannot answer from this context.
            # Trusting that is strictly better than forcing an answer.
            return refuse("model_reported_unanswerable", {"citations": citations})
        else:
            answer, mode = gen.answer, "generated"
    else:
        answer, mode = result.hits[0].context_text, "extractive"
        checks["degraded"] = {"reason": "no_api_key"}
        timings["generate"] = 0.0

    # --- gate 4: groundedness (post-generation, free, the real guardrail) ---
    t0 = time.perf_counter()
    grounded = g.check_groundedness(answer, contexts)
    timings["groundedness"] = round((time.perf_counter() - t0) * 1000, 3)
    checks["groundedness"] = {"ok": grounded.ok, "reason": grounded.reason, **grounded.detail}

    if not grounded.ok and mode == "generated":
        # A generated answer containing specifics absent from the context is
        # exactly the failure this system exists to prevent. Refuse it.
        return refuse("ungrounded_answer", {"citations": citations})

    timings["total"] = round((time.perf_counter() - t_start) * 1000, 3)
    return AskResponse(
        answer=answer,
        mode=mode,
        answered=True,
        citations=citations,
        guardrails=checks,
        timings_ms=timings,
    )


# Mounted last so API routes take precedence over the static index.
try:
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
except Exception:  # noqa: BLE001 - frontend is optional; the API stands alone
    log.warning("no static dir at %s; serving API only", STATIC_DIR)
