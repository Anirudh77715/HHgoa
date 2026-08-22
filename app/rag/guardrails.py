"""Guardrails: deciding when NOT to answer.

The retrieval benchmark established the constraint these are built around. On
48 labelled queries the answerable and must-refuse similarity distributions
OVERLAP -- at any score cutoff keeping >=95% of real questions, 6/10
must-refuse queries still get through. So no single gate works, and the design
is three cheap independent gates instead of one clever one:

  1. scope filter       pre-retrieval, catches imperatives ("write me a poem")
  2. score gate         pre-generation, a COST filter (blocks 4/10)
  3. groundedness       post-generation, the one that actually prevents
                        fabrication

All three are local, deterministic, and free -- no LLM-as-judge call. That
matters on a free tier where every generation call is rationed, but it also
makes them testable: the same input always produces the same verdict, so the
eval set measures the guardrail rather than sampling a model's mood.

Honest limits are documented per-check below. These catch fabricated specifics
(dates, amounts, names) which is the dominant hallucination risk for a corpus
made of dates, fees and rules. They do NOT catch a semantically wrong claim
built only from words that appear in the context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Imperative openers that signal a task request rather than a question about
# the event. Kept deliberately short: every entry is a phrase that cannot
# plausibly begin a genuine question about HH Goa, so false positives are
# near-impossible. A long keyword list would start refusing real questions.
IMPERATIVE_PATTERNS = [
    r"^\s*(write|compose|draft)\s+(me\s+)?(a|an|some)\b",
    r"^\s*(book|order|buy|reserve)\s+(me\s+)?(a|an|my)\b",
    r"^\s*(translate|summarize|rewrite|refactor)\s+",
    r"^\s*(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\b",
    r"^\s*(pretend|act as|roleplay|you are now)\b",
]

# Tokens worth verifying against the context. Fabricating one of these is the
# failure that actually misleads a user: a wrong date sends them on the wrong
# day, an invented fee makes them budget wrongly.
NUMBER_RE = re.compile(r"\b\d[\d,.:]*\b")
MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I
)
MONEY_RE = re.compile(r"[$₹€£]\s?\d[\d,.]*|\b\d+\s?(k|lakh|crore|usd|inr|rupees?)\b", re.I)
URL_RE = re.compile(r"https?://\S+|\b[\w.-]+@[\w.-]+\.\w+\b")

# Words a refusal legitimately contains; never treat these as fabrication.
HEDGE_WORDS = re.compile(
    r"\b(i (don't|do not) (have|know)|not (in|available|covered|specified|stated)|"
    r"no information|cannot (find|answer|confirm)|unable to)\b",
    re.I,
)


@dataclass
class Verdict:
    ok: bool
    reason: str
    detail: dict


def check_scope(query: str) -> Verdict:
    """Reject task requests and prompt-injection shaped input pre-retrieval.

    Catches: "write me a poem about hackathons", "book me a flight to Goa".
    Does NOT catch question-shaped out-of-scope input ("what is the weather in
    Goa in October") -- that is question-shaped and only groundedness can
    resolve it. Cheap first line, not a complete defence.
    """
    q = query.strip()
    if not q:
        return Verdict(False, "empty_query", {})
    if len(q) > 500:
        return Verdict(False, "query_too_long", {"length": len(q)})

    for pat in IMPERATIVE_PATTERNS:
        if re.search(pat, q, re.I):
            return Verdict(False, "out_of_scope_instruction", {"pattern": pat})
    return Verdict(True, "in_scope", {})


def check_vocabulary(term_hits: int, n_terms: int) -> Verdict:
    """Reject input sharing NO vocabulary at all with the corpus.

    WHY THIS EXISTS, and why it is not another threshold. A real voice test
    produced romanised Hindi, and `asdf qwer zxcv hjkl` was found to score
    cosine 0.8317 -- above the calibrated gate and above several genuine
    questions. An embedding always lands somewhere, so cosine cannot tell
    "unrelated language" or "keyboard mash" from "question": §4.2's overlap
    problem all over again.

    Lexical overlap can, and it separates cleanly:

        all 48 eval queries   min term-hits 2   (min maxBM25 2.81)
        nonsense / non-English      term-hits 0        maxBM25 0.00

    This is a BINARY STRUCTURAL TEST, not a tuned cutoff -- "does this query
    contain a single word the corpus has ever used?" -- so there is no number
    to overfit and no distribution to overlap.

    HONEST LIMITS:
      - Necessary, not sufficient. "goa asdf qwer" has one corpus term and
        passes. It catches total strangers, not partial ones.
      - Corpus-vocabulary bound, so a perfectly valid question in Devanagari or
        any other language is refused. That is a real limitation, not a fix --
        but refusing while SAYING the corpus is English-only beats answering it
        with a confidently wrong English passage, which is what happened before.
    """
    # Note there is deliberately NO `n_terms > 0` precondition. The tokenizer
    # matches [a-z0-9]+, so a query written entirely in Devanagari (or any
    # non-Latin script) yields ZERO terms -- and an earlier version treated
    # that as "nothing to check" and let it through. It only appeared to work
    # because the one test query happened to fall below the score gate as well;
    # a non-Latin query scoring above it would have been answered. Empty input
    # is already rejected by check_scope, so term_hits == 0 is decisive here.
    if term_hits == 0:
        return Verdict(
            False,
            "no_corpus_vocabulary",
            {"term_hits": 0, "query_terms": n_terms},
        )
    return Verdict(True, "vocabulary_overlap",
                   {"term_hits": term_hits, "query_terms": n_terms})


def check_score(top_score: float, threshold: float) -> Verdict:
    """Cost filter: skip the LLM call when nothing retrieved looks relevant.

    MEASURED LIMIT: blocks only 4/10 must-refuse queries at a cutoff that keeps
    38/38 answerable ones. Passing this gate is NOT evidence the answer exists
    in the corpus -- it only means we are not certain it doesn't.
    """
    if top_score < threshold:
        return Verdict(
            False,
            "below_retrieval_threshold",
            {"top_score": round(top_score, 4), "threshold": threshold},
        )
    # The threshold is reported on the PASS path too, not just the block path.
    # "confident (0.8615)" is unreadable without the bar it cleared, and a
    # reviewer checking whether the gate is doing anything needs both numbers.
    return Verdict(
        True,
        "retrieval_confident",
        {"top_score": round(top_score, 4), "threshold": threshold},
    )


def _checkable_tokens(text: str) -> set[str]:
    """Specifics whose fabrication would actually mislead someone.

    Applied to the ANSWER and to the CONTEXT with identical rules, so the two
    sides are compared as sets of normalised tokens rather than by substring.
    That symmetry is the point -- see check_groundedness.
    """
    toks: set[str] = set()
    for rx in (NUMBER_RE, MONEY_RE, URL_RE):
        toks |= {m.group(0).strip().lower() for m in rx.finditer(text)}
    # Months truncate to three letters on BOTH sides, so "Sept"/"September"
    # and "Oct"/"October" compare equal instead of near-missing each other.
    toks |= {m.group(0)[:3].lower() for m in MONTH_RE.finditer(text)}
    # Bare ordinals and single digits are noise ("1 of 3 things", "day 2").
    # MEASURED CONSEQUENCE: "The stake is 4 thousand rupees" yields no tokens
    # and is therefore not checked. Dropping the filter would flag ordinary
    # prose instead, so the noise is accepted and the gap documented.
    return {t for t in toks if len(t) > 1}


def check_groundedness(answer: str, contexts: list[str]) -> Verdict:
    """Verify every specific in the answer traces back to retrieved context.

    This is the gate that carries the system, because the score gate provably
    cannot. It is a NECESSARY-not-sufficient test:

      catches  invented dates, fees, counts, emails, URLs -- i.e. a made-up
               wifi password, a fabricated registration deadline, a prize
               amount the corpus never states
      misses   a claim that is semantically wrong but assembled only from
               tokens present in the context (e.g. attaching the right number
               to the wrong milestone)

    The residual risk is handled upstream instead: generation is forced to
    return cited chunk ids and an explicit `answerable` flag, so a model that
    cannot ground its answer has a way to say so rather than being cornered
    into inventing one.
    """
    haystack = " \n ".join(contexts).lower()
    ans_tokens = _checkable_tokens(answer)

    # A refusal contains no claims to ground.
    if not ans_tokens:
        if HEDGE_WORDS.search(answer):
            return Verdict(True, "refusal_no_claims", {"checked": 0})
        return Verdict(True, "no_checkable_claims", {"checked": 0})

    # Set difference over identically-extracted tokens, NOT `t in haystack`.
    # MEASURED: a substring test called a fabricated "50 builders" supported by
    # a context that only says "500 builders", and "7 May 202" supported by
    # "2026" -- false negatives in the one direction that matters, since a
    # missed fabrication is the failure this whole system exists to prevent.
    ctx_tokens = _checkable_tokens(haystack)
    unsupported = sorted(ans_tokens - ctx_tokens)
    if unsupported:
        return Verdict(
            False,
            "ungrounded_specifics",
            {
                "checked": len(ans_tokens),
                "unsupported": unsupported[:10],
                "n_unsupported": len(unsupported),
            },
        )
    return Verdict(True, "grounded", {"checked": len(ans_tokens)})


REFUSAL_TEXT = (
    "I don't have that in the Hacker House Goa event information I can see. "
    "Try asking about dates, eligibility, the selection process, tasks, or the terms."
)

# A vocabulary miss has a specific, actionable cause, so it gets its own text.
# "I don't have that" would be misleading for a question the corpus DOES cover
# but that was asked in a language the English-only index cannot represent.
NO_VOCAB_TEXT = (
    "I couldn't match any of that to the Hacker House Goa material. My source "
    "is the English text of hhgoa.com, so questions in other languages won't "
    "match even when the answer is there. Try asking in English — for example, "
    "\"when does registration begin\"."
)
