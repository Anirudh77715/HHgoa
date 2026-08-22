---
title: Ask HH Goa
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

<!-- The YAML block above is required by Hugging Face Spaces: without it a
     Docker Space refuses to build. GitHub renders it as a small table, which
     is a fair price for "push the same repo to either host and it just works".
     app_port must match the Dockerfile's EXPOSE/uvicorn port. -->

# Ask HH Goa — voice-enabled RAG

**Hacker House Goa 2026 · Task #2 · `#RAGInGoa`**

Speak a question about [hhgoa.com](https://hhgoa.com), get an answer grounded in
the site's own text — retrieved locally in **~6ms**, cited, and **refused
outright** when the corpus doesn't contain it.

The corpus is the event site itself. That was chosen deliberately: it's small
enough that every answer is hand-verifiable, so grounding claims can be checked
rather than asserted.

```bash
py -m uvicorn app.main:app --port 7860     # → http://localhost:7860
```

**No API key required.** Without one the app retrieves, ranks, cites and refuses
exactly as normal, and answers with the retrieved passage verbatim. A key only
upgrades the wording step. See [DEPLOY.md](DEPLOY.md).

---

## On the "under 200ms" requirement

A literal sub-200ms voice-to-spoken-answer **is not achievable**, and claiming it
would be a misunderstanding — LLM generation alone is 300ms–2s. So the pipeline
is reported per stage, and the UI shows this breakdown live:

```
you speaking                 (user time, not system cost)
speech recognition            ~420ms   browser service, uploads to Google
scope filter                    0.03ms
embed query                     5.9ms  ┐
vector search + BM25 fusion     0.3ms  ├─ retrieval subtotal ~6.5ms  ← the claim
collapse to parents             0.2ms  ┘
generation                    ~500ms   network, free-tier API
groundedness check              0.4ms
```

**"Retrieval: 6.5ms p50 over 2,880 timed calls"** is both more credible and more
impressive than an unqualified "<200ms".

## What's actually engineered

**Three chunking strategies, benchmarked** — not one naive split. 37 labelled
answerable queries + 11 that must be refused:

| strategy | chunks | recall@5 | prec@1 | MRR | p50 |
|---|---|---|---|---|---|
| **parent_child** | 91 | **0.919** | **0.676** | **0.763** | 5.96ms |
| structural | 32 | 0.892 | 0.622 | 0.737 | 5.90ms |
| recursive | 54 | 0.865 | 0.595 | 0.702 | 6.36ms |

Every chunk carries **two** texts: `embed_text` (what's searched — precision)
and `context_text` (what the model answers from — recall). Most naive RAG sets
these equal and forces one chunk size to serve two opposed jobs.

**Hybrid retrieval.** Dense search ranked *"Registration **Ends**"* above
*"Registration **Begins**"* for "when does registration begin" — a bi-encoder
cannot encode polarity. BM25 fused via RRF fixes it. The aggregate metric gain
is inside the noise floor and the README says so; the justification is the
failure mode, not the average.

**Four guardrails**, each catching what the previous cannot:

| gate | catches | cost |
|---|---|---|
| scope | imperatives, prompt injection | free, pre-retrieval |
| **vocabulary** | nonsense and other languages | free |
| score | obvious misses | free, pre-generation |
| groundedness | fabricated dates, fees, counts, URLs | free, post-generation |

The measured finding that shaped all of this: **a similarity threshold cannot be
the guardrail.** Over 48 labelled queries the answerable and must-refuse score
distributions *overlap* (separation −0.0447). At any cutoff keeping ≥95% of real
questions, 6 must-refuse queries still get through. The score gate is therefore
demoted to a cost filter, and groundedness does the real work.

The vocabulary gate is the counter-example worth noting: `asdf qwer zxcv hjkl`
scores cosine **0.8317**, above real questions. But it shares **zero** words with
the corpus, while all 48 eval queries share at least 2. That's a structural
discontinuity, not a tuned threshold — which is exactly why it works.

## Honest limitations

Documented rather than hidden, with the measurements behind them in
[CONTEXT.md](CONTEXT.md):

- **Voice input needs Chrome or Edge.** Web Speech is not on-device; it uploads
  audio to Google and only builds carrying Google's key are accepted. The page
  detects the browser and says so.
- **English only.** The embedding model is `bge-small-en`, so a valid Hindi
  question is refused — with a message saying the source is English-only, rather
  than answered with a confidently wrong English passage.
- **Groundedness verifies specifics, not semantics.** It catches invented dates
  and amounts; it does not catch a right number attached to the wrong milestone.
  Test case `g020` pins that blind spot open on purpose.
- **The 500-vs-247 builder-count conflict cannot yet be cited.** Retrieval
  returns one side but not the other, so the answer that reports the conflict is
  *refused* while the one that silently picks a winner passes — the inverse of
  what's wanted.

## Verify it yourself

```bash
py -m app.corpus.verify        # 14 corpus invariants
py -m bench.bench_retrieval    # recall@5, precision@1, MRR, P50/P70/P100
py -m bench.bench_hybrid       # dense vs BM25 vs fusion, bootstrap CIs
py -m bench.bench_guardrails   # guardrail acceptance suite
py -m bench.calibrate          # threshold sweep + the negative results
```

All local, all free, no API key. Committed reports in `bench/reports/`.

## Stack

Everything local except generation. Total running cost: **$0**.

`fastembed` (ONNX, no torch) · `BAAI/bge-small-en` · FAISS flat · hand-rolled
BM25 · FastAPI · Gemini 2.5 Flash-Lite · browser Web Speech + SpeechSynthesis

`bge-small-en` **not** `-v1.5`: fastembed ships the latter int8-quantized, and it
measured **152.70ms p50 against 6.73ms** on the same machine — a 23x trap from
picking the "obvious" default. Unfused quantize/dequantize nodes; ruled out
thread oversubscription and wrapper overhead with measurements first.

---

Full engineering log, including the negative results and the bugs found along
the way, is in **[CONTEXT.md](CONTEXT.md)**. Deployment in **[DEPLOY.md](DEPLOY.md)**.
