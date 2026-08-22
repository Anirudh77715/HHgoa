"""Runtime settings.

Deliberate design point: the API key is OPTIONAL. With no key the service still
answers, in extractive mode -- returning the best retrieved passage verbatim
instead of a generated paraphrase.

That is not a stub. It is the same code path used when the free-tier daily quota
runs out, so the degraded mode is exercised constantly rather than being
untested emergency code. It also means CI can health-check and integration-test
the whole retrieval path without secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- generation (optional) ---
    gemini_api_key: str | None = field(
        default_factory=lambda: (os.getenv("GEMINI_API_KEY") or "").strip() or None
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    )
    # Flash-Lite allows 30 RPM on the free tier vs 15 for Flash. We pace to the
    # limit with a token bucket rather than discovering it via 429s.
    gen_rpm_limit: int = field(default_factory=lambda: _int("GEN_RPM_LIMIT", 30))
    gen_timeout_ms: int = field(default_factory=lambda: _int("GEN_TIMEOUT_MS", 8000))
    gen_max_retries: int = field(default_factory=lambda: _int("GEN_MAX_RETRIES", 3))

    # --- retrieval ---
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
    )
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    # Calibrated on 48 labelled queries, not guessed. This is a COST filter that
    # avoids spending an LLM call on obvious misses -- it blocks only 5/11
    # must-refuse queries while keeping 37/37 answerable ones, so groundedness
    # does the real work. See bench/calibrate.py.
    retrieval_min_score: float = field(
        default_factory=lambda: _float("RETRIEVAL_MIN_SCORE", 0.8167)
    )
    # parent_child measured best recall@5 (0.919 vs 0.892 structural,
    # 0.865 recursive) AND best precision@1 (0.676), winning decisively on
    # paraphrased questions.
    strategy: str = field(default_factory=lambda: os.getenv("STRATEGY", "parent_child"))
    index_kind: str = field(default_factory=lambda: os.getenv("INDEX_KIND", "flat"))

    # dense | hybrid. Hybrid fuses BM25 with the dense ranking via RRF.
    #
    # This is NOT defaulted on for an average-metric win -- prec@1 moves 0.676
    # -> 0.703 (+1 query net of 37) and the bootstrap CIs overlap almost
    # entirely, so on aggregate the two are indistinguishable. It is on because
    # of an ASYMMETRY in the failure modes it trades:
    #
    #   fixed   "when does registration begin" returned *Registration Ends* at
    #           rank 1. Both passages then sit in context, so groundedness
    #           passes the wrong answer (guardrail case g020) -- a silent,
    #           confident error on the likeliest question this corpus gets.
    #   broken  3 queries lose rank 1 to a lexical false friend, but ALL THREE
    #           keep the gold doc inside the top 5, so generation still sees it.
    #
    # Degrading a rank is cheap; emitting a wrong answer no gate catches is
    # not. Set RETRIEVAL_MODE=dense to revert. See bench/bench_hybrid.py.
    retrieval_mode: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid").lower()
    )
    # RRF constant. 60 is the standard value from the original paper, kept
    # deliberately UNTUNED: sweeping it on the same 37 queries used to evaluate
    # is how you manufacture a result that does not generalise. A weighted
    # alpha=0.7 fusion scored higher (prec@1 0.757) but its immediate
    # neighbours 0.6 and 0.8 did not, which is the signature of a fluke rather
    # than an optimum.
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))

    # --- server ---
    # Comma-separated origins allowed to call the API cross-origin, or "*".
    # Needed because the frontend can be hosted apart from the API (Netlify
    # cannot run this Python backend -- see DEPLOY.md). Defaults to "*" because
    # this is a public, read-only, unauthenticated demo with no cookies and no
    # user data; the BYOK key travels in an explicit header the caller sets on
    # purpose, so there is no ambient credential for another origin to abuse.
    # Narrow it to your Netlify origin if you ever add state worth protecting.
    allowed_origins_raw: str = field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*")
    )
    port: int = field(default_factory=lambda: _int("PORT", 7860))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "info"))

    @property
    def allowed_origins(self) -> list[str]:
        raw = self.allowed_origins_raw.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def can_generate(self) -> bool:
        return self.gemini_api_key is not None

    @property
    def index_dir(self) -> Path:
        from app.index.embed import model_slug

        return Path("data/index") / model_slug(self.embed_model) / self.index_kind / self.strategy

    def redacted(self) -> dict:
        """Safe to log or return over HTTP: never exposes the key itself."""
        return {
            "gemini_model": self.gemini_model,
            "generation_enabled": self.can_generate,
            "embed_model": self.embed_model,
            "strategy": self.strategy,
            "index_kind": self.index_kind,
            "retrieval_mode": self.retrieval_mode,
            "rrf_k": self.rrf_k,
            "top_k": self.top_k,
            "retrieval_min_score": self.retrieval_min_score,
            "gen_rpm_limit": self.gen_rpm_limit,
        }


settings = Settings()
