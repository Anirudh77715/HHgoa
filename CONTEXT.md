# CONTEXT — Voice-Enabled RAG for HH Goa 2026

Working context for this project. Written so a fresh session (or another person)
can resume without re-deriving decisions or repeating dead ends.

**Last updated:** 2026-08-22 (attractor fix + precision@1 + guardrail suite +
hybrid retrieval + voice frontend)
**Repo:** `A:\Hackgoa` (git initialised, `main`, 3 commits, **not yet pushed**)

---

## 1. What this is

A submission for **Task #2** of [Hacker House Goa 2026](https://hhgoa.com/) —
a 4-day, 500-developer hackathon residency in Goa, 28–31 Oct 2026. This task is
a **qualifier/screening challenge**, not the hackathon itself. Organisers use it
to filter for people who can actually engineer a retrieval pipeline.

Task requirements, verbatim from the site:
- Speak the question — real voice-to-text, not typed
- Retrieval that's actually engineered — **multiple chunking strategies**, not one naive split
- Blazing-fast — "full pipeline under 200ms"
- **P50 / P70 / P100** latency, benchmarked across real queries
- Runs inside a real harness — retries, structured I/O, error recovery
- Guardrails that know when **not** to answer
- `#RAGInGoa` on every post

### Concept: "Ask HH Goa"

The RAG corpus **is hhgoa.com itself** — FAQs, schedule, task cards, terms.
A voice assistant answering "when do results come out", "is there a fee",
"what's task 2".

Chosen over a generic corpus because: small enough that every answer is
hand-verifiable (so grounding claims are trustworthy, not hand-waved), it forces
genuinely different chunking problems (FAQ pairs vs. prose vs. grid-structured
timeline), and it's directly useful to the organisers reading the submission.

### The honesty position on "under 200ms"

**A literal <200ms full voice-to-spoken-answer is not achievable** and claiming
it reads as a lie or a misunderstanding to any engineer judging it. LLM
generation alone is 300ms–2s.

Report a **staged breakdown** instead:

```
STT (browser)         ~250ms
embed (local)            7ms
FAISS search           0.1ms
guardrail                1ms
── retrieval subtotal   ~7ms   p50/p70/p100  <- the engineered claim
Gemini TTFT           ~400ms
full answer             ~1.2s
```

"Retrieval subsystem: 7.07ms p50 over 2,880 timed calls" is more credible *and*
more impressive than an unqualified "<200ms".

---

## 2. Current status

| Component | State |
|---|---|
| Corpus builder | ✅ working — 32 docs, 11,068 chars |
| 3 chunking strategies | ✅ working — 32 / 54 / 91 chunks |
| Local embeddings (ONNX) | ✅ working — 5.9ms p50 |
| FAISS index | ✅ working — exact flat, 0.08ms search |
| Labelled eval set | ✅ 48 queries (37 answerable, 11 must-refuse) |
| Retrieval benchmark | ✅ working — recall@5 + **precision@1** + MRR + attractors + P50/P70/P100 |
| `notice-board-00` attractor | ✅ removed from corpus + CI invariant (§8) |
| Guardrail calibration | ✅ working — with a critical negative finding |
| Scope filter | ✅ working (verified in smoke test) |
| Vocabulary gate | ✅ new — rejects nonsense / non-English cleanly, 0 false positives on 48 queries (§15) |
| Score gate | ✅ working (verified in smoke test) |
| FastAPI service | ✅ written, verified in-process (`/healthz`, `/config`, `/ask`) |
| Gemini generation | ⚠️ **written but BLOCKED — 401 on the API key** |
| Groundedness check | ✅ **tested — 18-case suite, 13/13 scored, 5 documented blind spots** (§4.6) |
| Guardrail suite in CI | ✅ runs with no secrets, fails only on undocumented regressions |
| Hybrid retrieval (BM25 + RRF) | ✅ default — fixes the begin/end antonym bug (§4.7) |
| `StaticFiles` mount | ✅ unblocked — `GET /` serves the app |
| Voice frontend | ✅ **complete — full spoken round trip confirmed in Chrome** (§14.3) |
| Cross-browser voice | ⚠️ Chrome/Edge only — decided and documented (§14.2) |
| Voice actually working | ✅ spoken question → cited answer → spoken reply, verified end to end (§14.3) |
| E2E benchmark | ❌ not started |
| Deployment (HF Spaces) | ❌ not started |

---

## 3. BLOCKERS

### 3.1 The Gemini API key is invalid — 401 UNAUTHENTICATED

Every generation call fails:

```
401 UNAUTHENTICATED
Request had invalid authentication credentials.
Expected OAuth 2 access token, login cookie or other valid authentication credential
```

The key loaded from `.env` is **53 characters**. Google AI Studio API keys are
**39 characters and start with `AIza`**. So this is not an AI Studio API key —
it's likely a Vertex AI / OAuth / service-account credential, or a wrong paste
(truncated, extra characters, or a different service entirely).

**Fix:** get a real AI Studio key at https://aistudio.google.com/apikey and put
it in `.env` as `GEMINI_API_KEY=`. It should look like `AIza...` and be 39 chars.

Note: the failure is handled *gracefully* — the harness classified it correctly
as `client_error`, did **not** retry (401 is non-retryable, retrying wastes
quota), and the service falls back to extractive mode. So the harness works;
only the credential is wrong.

**Correction — this blocker is narrower than previously recorded.** An earlier
version of this file listed the groundedness check as "untested, blocked by the
401". That was a misread dependency. `check_groundedness(answer, contexts)` is a
pure function of two strings and never touches the network; only the *end-to-end*
path needs a key. It is now tested — see §4.6. What genuinely remains blocked is
the E2E benchmark (§11 step 6) and observing what Gemini actually generates.

### 3.2 GitHub not authenticated

`gh` is installed (v2.97.0) but not logged in. Nothing has been pushed.

```bash
gh auth login
```

Also note: git identity is `anirudhm22 <anirudhm22@iitk.ac.in>` while the
session account is `satvik.singh@kissht.com`. Commits will be authored as
**anirudhm22** — fine if that's the GitHub account, otherwise fix per-repo.

### 3.3 Docker unavailable — deliberately worked around, do not retry

Windows 11 **Home** requires the WSL2 backend for Docker Desktop.
`wsl.exe` exists and the Store WSL app is installed (v2.2.4.0), but
**`LxssManager` is absent** — the underlying Windows optional features were
never enabled. So `wsl.exe` starts, finds no kernel, and reports "system cannot
find the file specified". `wsl --install` fails the same way unless elevated.

**Decision: do not fix.** It costs two elevated DISM calls and two reboots to
buy only fast local Dockerfile iteration. Replaced by
`.github/workflows/ci.yml`, which builds the image and health-checks the
container on every push — on clean Linux, which is better evidence the HF Spaces
build will work than a local build would be.

If ever needed: elevated PowerShell, `dism.exe /online /enable-feature
/featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`, reboot, then
`winget install -e --id Docker.DockerDesktop`.

---

## 4. Measured findings (the important part)

These are all backed by numbers in `bench/reports/`. Do not undo them.

### 4.1 The default embedding model was a 23x latency trap

`fastembed` ships `BAAI/bge-small-en-v1.5` as an **int8-quantized** ONNX build
(`bge-small-en-v1.5-onnx-q`). Measured on this machine (Ryzen 7 5800H, 8c/16t,
**249 GFLOPS** on fp32 matmul):

| model | p50 | p100 |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (int8) | **152.70ms** | 169.59ms |
| `all-MiniLM-L6-v2` | 22.01ms | 25.08ms |
| `BAAI/bge-small-en` (fp32) | **6.73ms** | 8.44ms |
| `snowflake-arctic-embed-xs` | 5.03ms | 8.57ms |

Ruled out, with measurements:
- **Thread oversubscription** — 1 vs 2/4/8/16 intra-op threads barely moved it
  (148ms → 168ms). Not the cause.
- **fastembed wrapper overhead** — raw `session.run` was *equally* slow (199ms),
  so the wrapper is innocent.
- **Fixed 512-token padding** — no; dynamic shapes work (4 chars 135ms vs
  2400 chars 960ms).

Cause: **unfused quantize/dequantize nodes**. Every op pays a
quant→compute→dequant round trip that dwarfs the actual matmul at this input
size. The "small and optimized" default would have silently eaten the entire
latency budget.

**Decision:** `EMBED_MODEL=BAAI/bge-small-en` (non-quantized). Model identity is
a benchmark axis, not a hardcoded guess.

### 4.2 A similarity threshold CANNOT be the guardrail

Over 48 labelled queries, the answerable and must-refuse score distributions
**overlap**:

```
answerable min  0.8207
refuse     max  0.8655
separation     -0.0447   <- negative = no cutoff can separate them
```

Operating-point curve (`bench/calibrate.py`, re-run after the §8 corpus fix):

| cutoff | answerable kept | refuse blocked | leaks |
|---|---|---|---|
| **0.8167** | **37/37** | 5/11 | **6**  <- shipped default |
| 0.8270 | 35/37 | 6/11 | 5 |
| 0.8345 | 33/37 | 7/11 | 4 |
| 0.8374 | 32/37 | 7/11 | 4 |

At any cutoff keeping ≥95% of real questions, **6 must-refuse queries get
through** — including "how many registrations were there in 2024", the exact
question the counter exclusion exists to prevent answering. Removing the §8
attractor did move the operating point (0.8183 → 0.8167, now blocking 5/11
instead of 4/10) but did **not** change the conclusion: the distributions still
overlap and the leak count is still 6.

Also note the **scale**: bge cosines cluster in 0.79–0.87. A "sensible-looking"
threshold like `0.42` (the initial guess) would gate *nothing*.

**Consequences, baked into the design:**
- Score gate demoted to a **cost filter** (avoid wasting an LLM call)
- **Groundedness check is mandatory**, not optional — it is the only thing
  preventing fabrication
- Calibration reports the **tradeoff curve**, not one "best" cutoff, because
  false refusals and fabrications are not symmetric costs

These 6 queries are the acceptance test for groundedness:
```
what is the weather like in Goa in October
write me a poem about hackathons          (caught by scope filter instead ✅)
book me a flight to Goa                   (caught by scope filter instead ✅)
how many registrations were there in 2024
what was the total bounty pool
who won the hackathon last year
```
`is anything pinned on the notice board` was added as an 11th must-refuse query
(§8) and is **blocked by the score gate**, so it is not on this list.

### 4.3 NEGATIVE RESULT — relative score signals are worse. Do not retry.

Hypothesis tested: a genuine match should *stand out* from the field, so
relative distinctness should separate better than absolute score.

**It does not.** AUC on the same 48 queries:

| signal | AUC |
|---|---|
| `top1` (absolute) | **0.900** |
| `spread` | 0.713 |
| `margin_1k` (top1 − mean rest) | 0.679 |
| `ratio_1k` | 0.657 |
| `margin_12` (top1 − top2) | 0.568 |

Kept in `bench/calibrate.py` so it isn't re-litigated.

### 4.4 parent_child wins recall, and wins where predicted

Over 37 answerable queries (2026-08-22, after the §8 corpus fix), 2,880 timed
calls per strategy:

| strategy | chunks | recall@5 | **prec@1** | MRR@5 | p50 | faq_paraphrase r / p1 |
|---|---|---|---|---|---|---|
| **parent_child** | 91 | **0.919** | **0.676** | **0.763** | 5.96ms | **1.00** / 0.29 |
| structural | 32 | 0.892 | 0.622 | 0.737 | 5.90ms | 0.71 / 0.43 |
| recursive | 54 | 0.865 | 0.595 | 0.702 | 6.36ms | 0.57 / 0.29 |

parent_child wins all three quality metrics. It wins decisively on **paraphrased**
questions by recall — small children match reworded queries that whole-doc chunks
miss. Exactly the predicted mechanism.

**But note the recall/precision gap:** 0.919 recall@5 against 0.676 prec@1. The
gold doc reaches the top 5 far more often than it reaches rank 1. Quoting recall@5
alone overstates the system by ~24 points, which is precisely how the §8 bug hid
for three commits. Report both.

Latency outlier: recursive shows p100 159.87ms against a 6.36ms p50 — one sample
in 2,880, an OS scheduling hiccup, not a property of the strategy (structural and
parent_child p100 are 8.6ms and 10.6ms on the same run). Worth stating rather than
quietly dropping, since p100 is reported as a true max.

**Nuance worth reporting:** the corpus has **two populations**. Short docs (FAQ,
timeline milestones) collapse to a single child, so parent-child degenerates to
"structural + title prefix" there; the 3.84x context amplification comes entirely
from the long terms clauses. A hybrid/router would likely beat all three.
(An earlier draft of this file said 2.44x. That figure is stale — it is not
explained by the §8 doc removal, which moves it by 0.01x.)

### 4.5 FAISS: exact flat on purpose

At **33–92 vectors**, an exact flat inner-product scan is ~35k multiply-adds —
microseconds, and *exact*. HNSW is approximate and *adds* graph-traversal
overhead, so it's strictly worse on both axes at this scale. HNSW stays
implemented (`--kind hnsw`) so the crossover can be *shown*, not asserted.

---

### 4.6 The groundedness check, actually measured

`bench/guardrail_cases.yaml` + `py -m bench.bench_guardrails`. 18 cases, real
contexts pulled live from the current index, no API key needed. Results:

```
scored cases      : 13/13 correct
documented limits : 5  (excluded from the score and listed explicitly)
undocumented fails: 0
```

The 13 scored cases include **five false-positive controls** — grounded answers
and well-formed refusals that must pass. A guardrail that blocks everything would
score 100% without them, so they are what make the number mean anything.

#### Bug found and fixed: substring matching

The check tested `token in haystack`. Any fabricated number that happened to be a
substring of a real one was reported as grounded:

| fabricated answer | real context | old verdict |
|---|---|---|
| "There will be **50** builders" | "**500** elite builders" | ✅ grounded (WRONG) |
| "Registration begins 7 May **202**" | "7 May **2026**" | ✅ grounded (WRONG) |

Both are false negatives in the one direction that matters — a missed
fabrication. Fixed by extracting tokens from the **context with the same rules**
and taking a set difference, so the two sides are compared as normalised tokens
instead of by substring. Months truncate to three letters on both sides, so
"Sept"/"September" still compare equal. Locked in as cases g030/g031; the five
false-positive controls confirm the stricter rule did not start over-blocking.

#### The 5 blind spots, measured not guessed

| case | what slips through | why |
|---|---|---|
| **g020** | "Registration begins on 1 October 2026" | every token IS in context — see below |
| g021 | "The weather is warm and humid at that time of year" | zero specifics, so nothing to verify |
| g022 | "The wifi password is HHGOA2026" | no word boundary before the digits, so `NUMBER_RE` never sees a number |
| g032 | "The stake is 4 thousand rupees" | `MONEY_RE` wants the unit adjacent; bare "4" is dropped as noise |
| **g023** | the *correct* conflict answer is **refused** — see below |

**g020 is the one to worry about — two bugs compose into a wrong answer nothing
catches.** Retrieval ranks `timeline-08` *Registration Ends — 1 October 2026*
**above** `timeline-00` *Registration Begins — 7 May 2026* (the antonym bug, §8).
Both passages are therefore in context, so "Registration begins on 1 October
2026" is fully token-supported and groundedness waves it through. The most likely
question this corpus will ever get can be answered confidently and wrongly, and
neither gate fires. Fixing this requires fixing *retrieval* (§8), not the
guardrail.

**g023 shows the conflict design failing from the other side.** §5 indexes the
500-vs-247 contradiction on purpose and the system prompt orders the model to
cite both figures. But retrieval for "how many builders attend" returns faq-00,
about-00, faq-01, faq-03, terms-00 — `about-00` (500) is there, `timeline-09` and
`terms-02` (247) are **not**. So an answer citing both is judged ungrounded on
"247" and refused, while the answer that silently picks 500 passes. Incomplete
retrieval inverts the intended behaviour exactly. The conflict cannot be reported
until retrieval returns both sides of it.

#### Why this is worth having

Three of the five blind spots (g021, g022, g032) are *coverage* gaps — the check
verifies specifics, not semantics, and says so in its own docstring. The other
two (g020, g023) are **retrieval failures wearing a guardrail costume**, which is
the useful discovery: no amount of work on `guardrails.py` fixes either one.

### 4.7 Hybrid retrieval fixes the antonym bug — but not on the averages

`py -m bench.bench_hybrid`. 91 chunks, 37 answerable queries, RRF and a weighted
sweep against dense-only.

| ranker | recall@5 | prec@1 | 95% CI (bootstrap) | MRR | paired vs dense |
|---|---|---|---|---|---|
| dense only | 0.946 | 0.676 | [0.514, 0.811] | 0.777 | — |
| bm25 only | 0.865 | 0.649 | [0.486, 0.784] | 0.733 | +4 / −5 = **−1** |
| **rrf k=60** (shipped) | 0.946 | **0.703** | [0.541, 0.838] | 0.800 | +4 / −3 = **+1** |
| rrf k=10 | 0.973 | 0.703 | [0.541, 0.838] | 0.809 | +4 / −3 = +1 |
| alpha=0.7 | 0.973 | 0.757 | [0.622, 0.892] | 0.841 | +4 / −1 = +3 |
| alpha=0.6 / 0.8 | 0.973 / 0.946 | 0.703 | [0.541, 0.838] | — | +1 |

**Read the CIs before the point estimates.** They overlap almost completely. On
37 answerable queries the whole visible spread is 1–3 queries, and *no fusion
setting here is statistically distinguishable from dense-only on aggregate*.

**alpha=0.7 is a trap and is NOT shipped.** It scores best on every metric, and
it was chosen as the peak of a sweep evaluated on the very queries it was tuned
on. Its immediate neighbours (0.6 and 0.8) both drop back to +1. A maximum its
neighbours do not share is the signature of a fluke, not an optimum. RRF k=60 is
shipped instead — the standard constant from the original paper, deliberately
untuned.

#### So why turn it on at all?

Not for the average. For an **asymmetry in what it trades**:

```
FIXED    q001  when does registration begin      -> timeline-00 now rank 1  ← the target
         q008  when are partner trials           -> timeline-06 now rank 1
         q019  can designers apply...            -> faq-00      now rank 1
         q021  is accommodation covered          -> faq-03      now rank 1

BROKEN   q006  when is RSVP and stake due        gold falls to rank 2
         q033  what is task 2 about              gold falls to rank 3
         q023  what is the minimum age           gold falls to rank 5
```

All three regressions **keep the gold doc inside the top 5**, so generation
still sees it — the cost is rank, not availability. The fix removes a case where
the model was being shown *Registration Ends* first and the guardrail could not
catch the resulting answer (§4.6 g020). Degrading a rank is cheap; emitting a
confidently wrong answer that no gate catches is not.

The regressions share a mechanism worth naming: **lexical false friends**. "what
is the minimum age to participate" has BM25 latch onto *participate*, which is
the headline word of faq-00 "Who can **participate**", while the real answer
(terms-01, "at least 18 years old") never uses the word "age". Dense handles
that; lexical doesn't. Exactly the complementarity that makes fusion worth
having, running in the other direction.

#### Two implementation points that matter

**Fusion reorders; it must not rescore.** RRF outputs sit on a reciprocal-rank
scale where the top result is roughly the same value no matter how good it is.
Gating on that would silently disable the score gate. So `RetrievalResult` now
carries `gate_score` — the best *dense cosine* over the candidate set, which is
the quantity `bench/calibrate.py` actually calibrated — separately from
`hits[0].score`, which fusion can lower. Verified live: q001 reports the
promoted chunk's cosine 0.8594 while gating on the set max 0.8615.

**No new artifact, no new dependency.** BM25 is built at load from the payloads
already in the index, so there is no extra build step and nothing extra on disk.
It is ~60 lines in `app/index/lexical.py` rather than `rank_bm25`, which keeps
the image lean and — more importantly — puts tokenisation under our control.
That matters: the query says "begin", the document says "Begins", and with no
suffix handling BM25 contributes *nothing* to the one case it was added for. A
crude 3-suffix stemmer handles it. It is not linguistically correct ("gas" →
"ga") but it is applied symmetrically to query and document, so a wrong stem
still matches itself.

**Cost:** retrieval p50 5.93ms → 6.92ms (+1.0ms), p100 9.7ms → 15.7ms. BM25
scoring itself is 0.06ms; the rest is the second argsort and fusion arithmetic
over 91 elements. Still an order of magnitude inside any sane budget.

#### What it does NOT fix

`q036` "how many builders attend" still returns `about-00` (500) without
`timeline-09` or `terms-02` (247) under every ranker tested. The 500-vs-247
conflict still cannot be cited (§4.6 g023). Fusion was never going to fix that —
it reorders candidates, and the missing passages are a recall problem.

## 5. Corpus engineering decisions

Source: 3 pages. Site is **Next.js but server-rendered** — all prose is in the
initial HTML, so no headless browser is needed.

**Never couple to CSS classes.** They're hashed Tailwind utilities that change
every rebuild. Anchor on semantic ids (`#notice-board`, `#tasks`, `#timeline`)
and heading text. The FAQ extractor has a text-pattern fallback that *logs when
it fires*, so markup drift degrades loudly instead of silently emptying the index.

### Three deliberate exclusions

1. **The four hero stat counters** (Registrations / Projects / Hackers /
   Bounties). They animate from 0 in client JS and the real values are **absent
   from server HTML**. A naive scraper indexes them as `0` and the system then
   confidently answers "0 hackers attended in 2024". Excluded → those queries
   must be refused. This is a real bug found in the site, not a contrived demo.

2. **`/result`** — a live leaderboard reveal. Answers go stale the moment a rank
   drops. Excluded as volatile (`--include-volatile` to override).

3. **Empty-state placeholders** — the notice board renders "Nothing pinned yet"
   until an organiser posts something. Volatile for the same reason `/result`
   is, *and* measurably the worst retrieval attractor in the index. Excluded by
   `EMPTY_STATE_RE` + a 60-char `MIN_DOC_CHARS` floor in `scrape.py`, both
   enforced as CI invariants. Full measurement in §8.

   The rule is **self-healing**: if a real notice is ever pinned, the doc stops
   matching the pattern and returns to the index on the next scrape with no
   code change.

4. Nothing else. See `data/corpus/manifest.json` — every drop is recorded there
   with its reason and the text that was dropped.

### The timeline pairing bug (nearly shipped broken)

The roadmap is a **positional CSS grid**: a dates column and an events column
paired only by index.

```
<div>                                          <- row group
  <div>  5x <div><span>date</span></div>       <- dates column
  <div>  5x <div><p>title</p><p>desc</p></div> <- events column
```

`get_text()` flattens this to
`"7 May 2026 August 2026 Early Sept 2026 ... Registration Begins Open Trials ..."`
— **destroying every date→event mapping**. Since "when does registration begin"
is the most likely question this corpus will ever get, that mapping *is* the
value of the section. Now paired index-wise, all 10 milestones recovered:

```
7 May 2026          -> Registration Begins
August 2026         -> Open Trials
Early Sept 2026     -> Alpha Selections
Early Sept 2026     -> Beta Selections
Mid Sept 2026       -> Charlie Selections
Mid Sept 2026       -> Delta Selections
September 2026      -> Partner Trials
Late September      -> RSVP & Stake
1 October 2026      -> Registration Ends
28–31 October 2026  -> Residency
```

### The 500-vs-247 conflict is INTENTIONAL

The site contradicts itself, 2-to-1:

```
about-00     "just 500 elite builders, high-speed fiber..."
timeline-09  "247 builders come together to build, ship, and launch..."
terms-02     "Residency — 28–31 Oct 2026 — 247 builders on-site in Goa"
```

Indexed on purpose. "How many builders attend?" is now a real test of whether
the system **cites conflicting sources** or silently picks a winner — far better
guardrail evidence than any question with a clean answer. The generation system
prompt explicitly instructs: *"If passages disagree, say so and give both
figures with their sources."*

### Corpus invariants (12 checks, enforced in CI)

`py -m app.corpus.verify`. Every check exists because violating it produces a
**confidently wrong answer**, not a visible crash: counter zeros must never
reappear; every timeline milestone must carry a date via *DOM pairing not a
fallback*; FAQs must keep question with answer; volatile docs must stay
unindexed.

---

## 6. Architecture — everything local except generation

Free-tier constraints pushed the design somewhere that is *also* better
engineering. Total running cost: **$0**, no trial clock.

| stage | choice | why not the API/paid option |
|---|---|---|
| voice→text | Browser **Web Speech API** (primary) + faster-whisper (bench) | zero quota, zero server load; whisper gives deterministic replay over fixed WAVs, which a browser API cannot. **But see §14.1 — it is a CLOUD service and only works in Google Chrome / Edge** |
| embeddings | **`bge-small-en`** local via **fastembed** (ONNX) | free *and* faster than any network call (6.7ms vs a round trip); no quota |
| vector search | **FAISS** flat, in-process | 0.1ms; a hosted vector DB adds a network hop for zero benefit |
| lexical search | **BM25**, hand-rolled, ~60 lines | fused with dense via RRF to fix the begin/end antonym bug (§4.7). No dependency, no artifact — built at load from the payloads already in the index |
| generation | **Gemini 2.5 Flash-Lite** | only external call. Flash-Lite over Flash: lower latency **and 30 RPM vs 15** |
| answer→voice | Browser **SpeechSynthesis** | free, instant, no server |
| guardrails | local, deterministic | no LLM-judge call → no quota burn, and *testable* (same input → same verdict) |
| host | **HF Spaces** (Docker, free CPU 2vCPU/16GB) | Render free tier sleeps after 15min with a 30–50s wake — disqualifying for a latency demo |

**Why `fastembed` not `sentence-transformers`:** same weights, ONNX Runtime
instead of PyTorch. Drops ~2GB of torch from the image and runs faster on CPU.

### Rate limits are a feature, not a problem

The task asks for "a real harness — retries, structured I/O, error recovery". On
a paid tier you'd have to invent a reason to demonstrate that. On the free tier
**429s are real**, so the harness genuinely needs: a token bucket pacing to
30 RPM (fail *less* rather than retry-storm), exponential backoff with jitter,
non-retryable classification (401/400/403 — retrying wastes quota; **this
already proved itself against the 401**), and graceful degradation to
**extractive mode**.

### Extractive mode is not a stub

With no key — or once quota is spent — the service returns the best retrieved
passage verbatim. **Same code path both times**, so the degraded mode is
continuously exercised rather than being untested emergency code. It also lets
CI integration-test the whole retrieval path with no secrets.

---

## 7. Smoke-test results (2026-08-22)

Retrieval and both free guardrails are **verified working**:

Re-run after the §8 fix, in-process via `fastapi.testclient`:

```
GET /healthz  -> 200 {"status":"ok","vectors":91,"generation_enabled":true}

Q: when does registration begin           top=0.8615  answered (extractive)
Q: is there a registration fee            top=0.8605  answered (extractive)
Q: how many builders attend               top=0.8444  answered (extractive)
Q: is anything pinned on the notice board -> REFUSED at score gate (0.8128 < 0.8167) ✅ NEW
Q: what is the wifi password              -> REFUSED at score gate (0.7964) ✅
Q: write me a poem about hackathons       -> REFUSED at scope filter ✅
Q: what is the capital of France          -> REFUSED at score gate (0.7828) ✅
Q: how many registrations in 2024         top=0.8655 -> PASSED score gate (the predicted leak) ⚠️
```

91 vectors. Generation: 401 on every call (see §3.1), so every answer above fell
back to **extractive mode** — which is the degraded path working exactly as
designed. Groundedness is not exercised on extractive answers (they are grounded
by construction), so it remains untested until the key is fixed.

---

## 8. FIXED — the `notice-board-00` attractor, and what it was hiding

### The bug

`notice-board-00` was a 41-char doc ("Pinned Up Notice Board Nothing pinned yet")
that surfaced as **top-1 for unrelated queries**. `recall@5` stayed 0.921 because
gold docs still appeared *somewhere* in the top 5, so the benchmark's only
quality metric **could not see it**.

### Cause — measured, not assumed

Mean cosine of every chunk against six deliberately unrelated control queries
(car tyres, boiling point of water, cookie recipes, Moby Dick, photosynthesis,
yen exchange rate):

```
notice-board-00 ranked  1 / 92  by mean cosine against the 48 corpus queries
notice-board-00 ranked  1 / 92  by mean cosine against the 6 CONTROL queries (0.7672)

corr(embed_text length, mean cosine over corpus queries)  = -0.450
corr(embed_text length, mean cosine over control queries) = -0.508
```

Ranking #1 against content it shares *no topic with* is the definition of a
centroid attractor, and rules out "it is actually relevant". The negative
correlation — **stronger for unrelated queries than related ones** — is the
mechanism: short texts land near the centroid of embedding space, so they score
moderately well against everything.

### Fix

Corpus-level exclusion in `scrape.py`, applied once over all docs so a new
extractor cannot forget it:

- `EMPTY_STATE_RE` — "nothing pinned yet", "coming soon", "no updates", …
- `MIN_DOC_CHARS = 60` — the shortest genuine doc on the site is a 95-char
  timeline milestone, so the floor cannot reach real content

Both are enforced as **CI invariants** in `verify.py` (now 14 checks, was 12), so
a re-scrape cannot quietly reintroduce one. Every drop is printed and recorded in
`manifest.json`.

`bench/queries.yaml` q038 ("is anything pinned on the notice board") was
reclassified `answer` → `refuse`, category `volatile`. The eval set is now
**37 answerable / 11 must-refuse**. The score gate blocks it correctly (§7).

### The honest result: precision@1 did NOT improve

```
                    before          after
recall@5            0.921           0.919
precision@1         0.684 (26/38)   0.676 (25/37)
MRR@5               0.763           0.763
```

Both figures moved only by the removal of q038, which was itself a correct
prec@1 hit. **On the 37 queries common to both runs, precision@1 is identical.**

Exact counterfactual (flat inner-product is exact, so re-inserting one vector
cannot perturb any other score — no rebuild needed to reproduce the old ranking):

| strategy | rank-1 stolen from **answerable** | rank-1 taken on **must-refuse** |
|---|---|---|
| structural | 7 | 7 |
| recursive | 4 | 6 |
| **parent_child** (shipped) | **3** (q001, q007, q028) | 4 |

All three of those answerable queries are **still** prec@1 misses after removal,
because the next-ranked doc is also wrong. The attractor was sitting on top of a
second layer of retrieval failures and masking them.

So the fix is worth keeping — it removes volatile content, kills the worst
attractor, and stops 4 must-refuse queries anchoring on a placeholder — but
**it bought no measured precision@1**. Do not report it as a precision win.

### What it was hiding — the real top finding

Full prec@1 miss list for parent_child (12 of 37):

```
q001  when does registration begin       gold timeline-00  got timeline-08  rank 2
q008  when are partner trials            gold timeline-06  got terms-02     rank 2
q009  what comes after beta selections   gold timeline-04  got timeline-03  rank 3
q016  do I have to pay anything to join  gold faq-03       got terms-03     rank 3
q019  can designers apply...             gold faq-00       got faq-05       rank 2
q020  will you help me find teammates    gold faq-04       got about-00     rank 4
q021  is accommodation covered           gold faq-03       got terms-07     rank 4
q022  how big can my team be             gold faq-00       got faq-04       rank 4
q036  how many builders attend           gold about-00     got faq-00       rank 3
q007  what dates is the hackathon        gold timeline-09  got about-00     NOT IN TOP 5
q028  how do I contact the organizers    gold terms-12     got faq-02       NOT IN TOP 5
q037  who organizes hacker house goa     gold terms-00     got faq-00       NOT IN TOP 5
```

**q001 is the worst of these and the most important question this corpus will
ever get.** "when does registration begin" returns `timeline-08` —
***Registration Ends*** — at rank 1, with `timeline-00` *Registration Begins* at
rank 2. A bi-encoder embedding cannot separate `begin` from `end`: the two docs
are near-identical apart from one antonym, and cosine similarity does not encode
negation or polarity.

This was a genuinely harder problem than the attractor. It is **not** fixable by
another corpus exclusion.

**✅ FIXED 2026-08-22 by hybrid BM25+RRF retrieval — see §4.7 for the full
measurement.** Lexical scoring separates "begin" from "ends" exactly where the
embedding cannot; `timeline-00` is now rank 1. Two caveats recorded there and
worth repeating: the aggregate metric gain is inside the noise floor (the
justification is the failure mode, not the average), and the *guardrail* blind
spot is unchanged — `timeline-08` is still in the top 5, so groundedness would
still pass "Registration begins on 1 October 2026" if a model produced it
(§4.6 g020). Only the likelihood dropped.

A cross-encoder rerank over the top 5 remains the next option if more is needed,
at a latency cost against the one budget this project actually claims.

3 queries (q007, q028, q037) have gold **outside the top 5 entirely**, so they
are recall failures, not ranking failures, and a reranker cannot help them.

### Residual attractor risk

With `notice-board-00` gone, no doc dominates. The remaining top-1-while-not-gold
counts are all at the `ATTRACTOR_MIN = 3` reporting floor (parent_child: faq-02,
faq-00, tasks-01 — 3 queries each out of 48), consistent with ordinary topic
overlap rather than a centroid effect.

Not addressed: the **chunk** floor. `MIN_DOC_CHARS` guards docs, not chunks, and
`recursive` still produces a 20-char chunk. It is not currently an attractor
(recursive's offenders are all full-length docs), so it is logged here rather
than fixed speculatively.

## 9. File map

```
A:\Hackgoa\
├── CONTEXT.md                    <- this file
├── .env                          <- GEMINI_API_KEY (gitignored; currently INVALID)
├── .env.example                  <- documents every setting + why
├── .gitignore                    <- secrets ignored before any key existed
├── Dockerfile                    <- HF Spaces: port 7860, uid 1000, model baked in
├── requirements.txt              <- runtime (torch-free)
├── requirements-bench.txt        <- + faster-whisper, matplotlib (not deployed)
├── .github/workflows/ci.yml      <- replaces local Docker
├── app/
│   ├── config.py                 <- settings; API key is OPTIONAL by design
│   ├── main.py                   <- FastAPI: /healthz /config /ask + static mount
│   ├── static/
│   │   └── index.html            <- voice UI; hhgoa.com palette, no build step (§14)
│   ├── corpus/
│   │   ├── scrape.py             <- 3 pages -> 33 docs; exclusions documented
│   │   └── verify.py             <- 12 invariants, CI-enforced
│   ├── chunking/
│   │   ├── strategies.py         <- structural | recursive | parent_child
│   │   └── build.py              <- builds all 3 + comparison stats
│   ├── index/
│   │   ├── embed.py              <- fastembed; the 23x finding is documented here
│   │   ├── lexical.py            <- BM25 + crude stemmer; no dependency, no artifact
│   │   ├── store.py              <- FAISS flat/hnsw; flat rationale documented
│   │   └── build.py              <- one index per (model, kind, strategy)
│   └── rag/
│       ├── retrieve.py           <- overfetch + collapse to unique parents + RRF fusion
│       ├── guardrails.py         <- scope | score | groundedness
│       └── generate.py           <- Gemini + token bucket + backoff + schema
├── bench/
│   ├── queries.yaml              <- 48 labelled queries w/ gold doc_ids
│   ├── guardrail_cases.yaml      <- 18 groundedness cases + declared blind spots
│   ├── bench_retrieval.py        <- recall@5 + precision@1 + MRR + attractors + P50/P70/P100
│   ├── bench_guardrails.py       <- guardrail acceptance suite (no API key needed)
│   ├── bench_hybrid.py           <- dense vs bm25 vs fusion, with bootstrap CIs
│   ├── calibrate.py              <- threshold sweep + the negative result
│   └── reports/                  <- committed JSON reports
└── data/
    ├── corpus/{docs,manifest}.json
    ├── chunks/{structural,recursive,parent_child,summary}.json
    └── index/BAAI__bge-small-en/flat/{strategy}/
```

### Key design point: `embed_text` vs `context_text`

Every chunk carries **two** texts:
- `embed_text` — what gets embedded and searched (**precision**)
- `context_text` — what the LLM answers from (**recall**)

Most naive RAG sets these equal, forcing one chunk size to serve two opposed
jobs. Splitting them is what makes parent-child possible at all.

### `retrieve.py` subtlety

With parent_child, several top-k **children** often share one parent. Returning
them raw looks like 5 results but is 1 context repeated 5 times — wasting the
prompt and making thin retrieval look well-supported. So we **overfetch 4x then
collapse to unique parents** (best child wins). `k` means "k distinct source
documents".

---

## 10. Commands

```bash
# corpus
py -m app.corpus.scrape          # rebuild from live site
py -m app.corpus.verify          # 12 invariants

# chunk + index
py -m app.chunking.build
py -m app.index.build --model "BAAI/bge-small-en"

# benchmark
py -m bench.bench_retrieval --model "BAAI/bge-small-en" --repeats 60
py -m bench.calibrate --model "BAAI/bge-small-en" --strategy parent_child

# guardrail acceptance suite (deterministic, no API key, runs in CI)
py -m bench.bench_guardrails --verbose

# dense vs bm25 vs fusion, with bootstrap CIs on precision@1
py -m bench.bench_hybrid

# serve  (UI at http://localhost:7860, API at /ask)
py -m uvicorn app.main:app --port 7860
```

Venv is at `A:\Hackgoa\.venv` (Python 3.12.10). Use
`.\.venv\Scripts\python.exe`. For scripts outside the repo, set
`PYTHONPATH=A:\Hackgoa`.

---

## 11. Next steps, in order

1. **Fix the API key** (§3.1) — blocks everything downstream
2. ~~Fix `notice-board-00` attractor + add `precision@1`~~ ✅ done 2026-08-22 (§8)
3. ~~Test the groundedness check against the known leaks~~ ✅ done 2026-08-22 (§4.6).
   Found and fixed a substring-matching bug; catalogued 5 blind spots; the suite
   now runs in CI with no secrets.
3b. ~~Fix the begin/end antonym failure~~ ✅ done 2026-08-22 (§4.7). Hybrid
   BM25+RRF retrieval is now the default; q001 ranks correctly. Note the
   guardrail blind spot itself is unchanged — only the probability of hitting
   it dropped (g020 stays a documented limit and says why).
3c. **Make retrieval return both sides of the 500-vs-247 conflict** (§4.6 g023).
   Today the conflict-citing answer is *refused* and the silently-wrong one
   passes — the exact inversion of what §5 set out to build. Fusion does not
   help; this is a recall problem, so try raising `k` for conflict-shaped
   queries, or indexing an explicit cross-reference.
4. ~~Voice frontend~~ ✅ done 2026-08-22 (§14). **One manual step left: test
   the microphone by hand in Chrome/Edge** — the automation browser blocks
   device capture, so the Web Speech path is unverified.
4b. ~~Decide on a server-side STT fallback~~ ✅ decided 2026-08-22: **browser
   Web Speech only**, limitation documented (§14.2). Revisit only if the
   Diagnose run below shows browser STT cannot be made to work at all.
4c. ~~Confirm voice works in a genuine Chrome window~~ ✅ done 2026-08-22 —
   speech service reachable, mic opens, voices present (§14.1, §14.1b).
4d. ~~Complete one spoken round trip in Chrome~~ ✅ done 2026-08-22 (§14.3).
   The waveform's concurrent mic stream turned out NOT to interfere; `?wave=0`
   stays available as a diagnostic escape hatch.
5. ~~Run the server, verify `/healthz`~~ ✅ done — boots in ~360ms, `GET /`
   serves the UI, `/healthz` returns 91 vectors
6. **E2E benchmark** — `bench/bench_e2e.py`: 40 queries × 5 runs ≈ 200 Gemini
   calls ≈ 13% of the 1,500/day free quota, ~7 min at 30 RPM
7. **faster-whisper** server path for deterministic benchmark replay
8. **Deploy to HF Spaces**, README with architecture diagram + latency table
9. **Post with `#RAGInGoa`**

### Cut list — do not build these

Multi-user auth · fine-tuning · Kubernetes · arbitrary document upload ·
multi-language · admin dashboard · cloud autoscaling · hosted vector DB ·
Gemini embeddings (slower than local) · LLM-as-judge guardrail (burns quota,
non-deterministic) · UI polish beyond functional

---

## 12. Things NOT to redo

- **Don't switch to `bge-small-en-v1.5`** — it's the int8 build, 23x slower (§4.1)
- **Don't try relative score signals** for the guardrail — measured worse (§4.3)
- **Don't raise `RETRIEVAL_MIN_SCORE`** hoping to fix refusals — the
  distributions overlap; it cannot work (§4.2)
- **Don't use HNSW** at this corpus size — flat is exact *and* faster (§4.5)
- **Don't install Docker/WSL** — CI covers it (§3.3)
- **Don't couple scrapers to Tailwind classes** — hashed, change every build (§5)
- **Don't claim <200ms end-to-end** — report the staged breakdown (§1)
- **Don't index the hero counters, `/result`, or empty-state placeholders** —
  they poison answers (§5, §8)
- **Don't report the §8 attractor fix as a precision@1 win** — measured: prec@1
  was unchanged on the common queries. It removed volatile content and exposed
  the real failures; that is the honest claim (§8)
- **Don't quote recall@5 on its own** — it overstates this system by ~24 points
  against precision@1, and that gap is exactly what hid the §8 bug (§4.4)
- **Don't claim the groundedness check needs an API key to test** — it is a pure
  function; only the E2E path is blocked (§3.1, §4.6)
- **Don't use substring matching to verify tokens** — it called "50" supported by
  "500". Compare token sets extracted identically from both sides (§4.6)
- **Don't quote the guardrail score without the blind spots** — 13/13 is only
  meaningful alongside the 5 declared limits and the 5 false-positive controls
  (§4.6)
- **Don't ship `alpha=0.7` fusion** — best on every metric, and selected as the
  peak of a sweep evaluated on its own tuning set. Neighbours 0.6/0.8 don't
  share the peak. RRF k=60 is untuned and honest (§4.7)
- **Don't gate on a fused score** — RRF is scale-free, so the score gate would
  silently stop working. Gate on `gate_score` (best dense cosine) (§4.7)
- **Don't claim hybrid is a quality win on the averages** — the bootstrap CIs
  overlap almost entirely. The justification is the asymmetric failure mode,
  and that is the claim to make (§4.7)
- **Don't describe browser STT as "local" or "offline"** — it uploads audio to
  Google and needs Chrome/Edge specifically (§14.1)
- **Don't feature-detect `SpeechRecognition` or `speechSynthesis` and assume
  they work** — both exist and both fail (one loudly, one silently) in
  Chromium/Electron builds (§14.1)
- **Don't read `getVoices()` synchronously** — it is async and returns `[]` on
  first call in Chrome. Gating `speak()` on it broke TTS in the one browser
  where it actually works (§14.1b)
- **Don't blame the network for a Web Speech `network` error** until the engine
  badge says `Google Chrome` — same machine, same network, Chrome passed and a
  Chromium fork failed (§14.1)
- **Don't time STT from `recog.start()`** — that measures the user talking, not
  the recognizer. Use `onspeechend` → final result (§14.3)
- **Don't re-sort citations by cosine to make them look ordered** — hybrid RRF
  sets the order and the cosine is reported unmodified. Tidy display, dishonest
  ranking (§14.3)
- **Don't verify UI from screenshots alone** — the latency bars were 0×0 for
  their entire life and looked fine at a glance. Assert on
  `getBoundingClientRect()` (§14.3)
- **Don't reach for a similarity threshold to reject nonsense** — `asdf qwer
  zxcv hjkl` scores cosine 0.8317, above real questions. Use lexical overlap,
  which has an actual discontinuity at zero (§15)
- **Don't guard a check with `if n_terms and ...`** — non-Latin scripts tokenize
  to zero terms and skip the check entirely (§15)

---

## 13. Commit history

```
ca9691d  Add embedding, FAISS retrieval, and a labelled benchmark
2d12aa5  Add chunking strategies, corpus invariants, and CI
bd57625  Scaffold voice RAG project and build hhgoa.com corpus
```

Uncommitted at time of writing: `app/config.py`, `app/main.py`, `app/rag/*`,
`CONTEXT.md`.

Pending cleanup (blocked by a protective hook on this directory — run manually):
```bash
git -C A:\Hackgoa rm -r data/index/flat data/index/build-flat.json
```
Those are stale artifacts from the first build, containing vectors from the
153ms quantized model.

---

## 14. Voice frontend

`app/static/index.html` — one self-contained file. No framework, no CDN script,
no build step. The only remote request is Google Fonts, with a full fallback
stack so a blocked font request degrades to system mono/serif rather than
breaking the page.

Serving it also unblocks the `StaticFiles` mount in `main.py`, which had been
logging `no static dir at app/static; serving API only` on every boot.

### Design is taken from hhgoa.com, measured not eyeballed

The palette and type were read off the live site's **computed styles**, not
guessed from screenshots:

```
background   #0B6839  deep green      body copy   #FFFFFF
accent 1     #FEE101  yellow          panels      #FFFBE8  cream
accent 2     #FF0080  hot pink
display font Imbue         700/900, uppercase
body font    Victor Mono   400/500/700
```

The site's primary CTA is square-cornered yellow with a soft yellow glow
(`0 0 16px 3.5px rgba(254,225,1,.27)`); that is reused for the mic button, with
pink reserved for stop/refuse states. Sections alternate green ground and cream
panels, as the site does.

### The interactive panels are the evidence, not decoration

Three animated panels exist because they make the scored requirements legible
at a glance:

| panel | what it shows | why it earns its place |
|---|---|---|
| **pipeline** | scope → retrieve → score → generate → grounded, each node lighting pass / stop / skipped | "Guardrails that know when not to answer" is scored. A refusal renders as a first-class result naming the gate that fired — never an error state |
| **confidence** | top cosine vs the calibrated gate, on the real 0.70–0.95 scale | A 0–1 bar would squash every result into a stripe and hide the population overlap that is the whole reason the score gate is only a cost filter (§4.2) |
| **latency** | proportional bars per stage, retrieval subtotal highlighted | The staged breakdown from §1, made visual. A **failed** generation call still shows its full network cost |

Also: typewriter answer reveal, frequency-bar waveform (yellow, going pink on
peaks), citations as pink-ruled rows carrying `doc_id` and cosine.

### 14.1 The Web Speech API is not local, and not portable

**Observed 2026-08-22:** `Speech recognition error: network`.

The cause is not a bug in the page. Chrome's `SpeechRecognition` is **not
on-device** — it uploads the audio to Google's speech backend, and that endpoint
only accepts requests from builds carrying Google's API key. In practice that
means **Google Chrome and Microsoft Edge only**. Plain Chromium, Electron
shells, Brave and most forks expose the `SpeechRecognition` interface and then
have the request rejected, which surfaces as a bare, misleading `network` error.

Measured in the Claude Code browser pane:

```
ua      Mozilla/5.0 … Claude/1.34493.1 Chrome/148.0.7778.280 Electron/42.9.2 …
brands  [ "Not/A)Brand", "Chromium" ]      <- no "Google Chrome"
hasSR   true        the interface EXISTS
voices  0           speechSynthesis has no voices either
```

Two traps here, and they are the same trap twice:

1. **Feature detection is not enough.** `window.SpeechRecognition` being present
   proves nothing; the object exists and still cannot work.
2. **`speechSynthesis` fails silently.** `getVoices()` returns an empty list, so
   `speak()` succeeds and produces no sound. A voice demo that is quietly mute
   is worse than one that admits it cannot speak.

**RESOLVED 2026-08-22.** Two Diagnose runs on the same machine settled it:

```
chromium-like build                     real Chrome
-------------------------------------   -------------------------------------
FAIL  browser engine — chromium-like    PASS  browser engine — chrome
PASS  microphone opens — AMD Array      PASS  microphone opens — AMD Array
PASS  google.com reachable              PASS  google.com reachable
FAIL  bare recognizer — error:network   PASS  bare recognizer — no error in 7s
```

Identical machine, identical network, identical microphone, **opposite result**.
That isolates the cause completely: it is the browser build, not the network.
`google.com reachable` passing in the failing run rules out a proxy or firewall
outright. **Browser STT works here in Chrome.**

Note the failing build was *not* the Electron pane in that run — it reported a
working mic and 6 voices, so it was a separately installed Chromium fork
(Brave/Opera/Vivaldi-class). Two different non-Chrome Chromium builds, same
`network` rejection. The rule generalises: Google Chrome and Edge only.

So the §14.2 decision stands, now on evidence rather than assumption. If a
future `network` failure ever appears *with the badge reading Google Chrome*,
these are the candidates to chase (and only then):

- a corporate proxy, VPN or DNS filter blocking Google's speech endpoint
- an extension (uBlock Origin, Privacy Badger, a strict blocklist)
- a captive portal, or an offline/flaky moment
- microphone permission never actually granted, so the service is never reached
  (this reports as `not-allowed`, but people read it as "voice is broken")
- the page's own second mic stream (the waveform) disturbing recognition

**So the page now identifies its own browser, permanently.** A pill under the
mic reads `VOICE READY` / `VOICE UNAVAILABLE` plus `running in <build>` on every
load. Which build is running the page decides whether voice can work at all, and
burying that inside an error message nobody reads until they are already stuck
cost a full debugging round trip.

The **Diagnose** button then isolates each remaining cause and prints a verdict:

```
PASS  browser engine — chrome
PASS  secure context — https://…
PASS  navigator.onLine
PASS  SpeechRecognition present
PASS  speechSynthesis voices — 12 voices
PASS  microphone opens — "Headset Microphone (Realtek)"
FAIL  google.com reachable — blocked by network, proxy or extension
FAIL  bare recognizer, en-US, no waveform — error:network
```

Two design points that make it actually diagnostic rather than decorative:

1. **The bare-recognizer probe runs with no concurrent waveform stream and the
   default language**, so it removes the two variables this page introduced.
   If the bare probe succeeds and the main button fails, the waveform is the
   culprit — and `?wave=0` disables it without touching code.
2. **Results are three-way, not pass/fail.** `not-allowed` and `audio-capture`
   mean the mic never opened, so the speech service was never contacted; those
   report as INCONCLUSIVE. Scoring them as "service reachable" would send you
   chasing entirely the wrong cause, which is exactly the mistake the original
   bare `network` message caused.

**What the page does now:**
- checks `navigator.userAgentData.brands` for "Google Chrome"/"Microsoft Edge"
  and warns **before** the first click rather than failing on it
- retries `network` exactly once (it is genuinely transient sometimes), then
  stops — with no API key it will fail identically forever
- gives a specific message per error code, plus a diagnostic line
  (`engine=… · secure=… · online=… · origin=…`)
- reports zero-voice `speechSynthesis` instead of playing silence
- focuses the typed box, which exercises the identical retrieval pipeline

### 14.1b Self-inflicted bug: `getVoices()` is asynchronous

The first Diagnose run in real Chrome reported `speechSynthesis voices — 0
voices`, which looked like a second, separate portability problem. It was not.
It was this page's bug, and a damaging one.

`speechSynthesis.getVoices()` is **populated asynchronously**. Chrome returns an
empty array on the first synchronous call and fills the list moments later,
firing `voiceschanged`. Both the diagnostic and `speak()` read it synchronously
at load, so both saw zero.

The diagnostic misreporting was cosmetic. The `speak()` bug was not:

```js
if (!synth.getVoices().length) { report("playback unavailable"); return; }   // WRONG
```

That guard was added to catch silent TTS failure in Electron — and it *caused*
exactly the failure it was written to prevent, refusing to speak in real Chrome
where voices were about to become available. A guard that turns a working path
into a broken one is worse than no guard.

**Fixed two ways:**
- `waitForVoices()` awaits `voiceschanged` with a polling fallback (some engines
  never fire the event) and a timeout; the diagnostic awaits it before reporting.
  The Electron pane now reports **5 voices**, not 0 — so even there the original
  reading was wrong.
- `speak()` **no longer gates on the voice list at all.** It attempts playback
  and detects *real* silence instead: if `onstart` never fires and the synth
  reports neither `speaking` nor `pending` after 1.5s, then it reports. Verified:
  `speaking: true` after an answer, no false warning.

General lesson, and the second time this exact shape has bitten on this page
(§14.1): **for Web Speech, presence of an API says nothing about it working, and
absence of data may just mean it has not arrived yet.** Probe behaviour, not
capability flags.

### 14.2 DECISION: Chrome/Edge only, documented

Judges will open the deployed Space in whatever browser they use. Real voice
input is an explicit, scored task requirement. As it stands, **the voice path
works in Chrome and Edge and nowhere else**, and there is nothing the frontend
can do about that.

The durable fix is a **server-side STT fallback**: `MediaRecorder` in the
browser → POST the audio → `faster-whisper` on the server. Every browser can
record audio; none of them need Google's key.

The cost is real and cuts against a stated project value:

| | browser Web Speech | server faster-whisper |
|---|---|---|
| works in | Chrome, Edge only | any browser with `MediaRecorder` |
| latency | ~250ms | ~300–800ms CPU on `tiny`/`base` |
| server load | none | real CPU per request on a 2-vCPU free Space |
| image size | +0 | **+~500MB–1GB** (CTranslate2 + model weights) |
| offline | no (uploads to Google) | yes, fully local |

`faster-whisper` is currently in `requirements-bench.txt` and deliberately NOT
deployed, precisely to keep the image lean (§6).

**DECIDED 2026-08-22: keep browser Web Speech only, and state the limitation
plainly in the README.** Zero image cost, ~250ms STT, and the page now warns
up front rather than failing mysteriously.

**This decision is contingent on one thing:** browser STT has to work *somewhere*
for the author, or there is no demo and no video. It reportedly fails in real
Chrome on the current machine/network, so run the **Diagnose** button in Chrome
first (§14.1). If the blocker turns out to be a corporate network or an
extension, a hotspot or an incognito window resolves it and the decision stands.
If browser STT genuinely cannot be made to work, revisit — the server-side
Whisper path above becomes necessary rather than optional, image size or not.

### Bugs the browser found that the tests did not

Running it surfaced four real defects that no unit test would have caught:

1. **`score — retrieval_confident (0.8615 vs undefined)`.** `check_score`
   attached `threshold` only to the *blocking* verdict, so the passing one had
   no bar to report against. Fixed in `guardrails.py` — both paths now carry it.
2. **The biggest number on the page was invisible.** The server records
   generation timing under `generate`; the UI read `generation`. A *failed*
   Gemini call still burns its whole ~300–900ms round trip, and it was being
   swallowed into "server total". On a page whose entire argument is an honest
   latency breakdown, that was the worst possible thing to hide.
3. **The threshold marker was clipped** by `overflow:hidden` on the meter, so
   the one number that explains the refusal never rendered.
4. **Mobile:** the pipeline's wrapped row had no right edge (`:last-child`
   only closes the final node — now every node self-closes with a −1px margin),
   and an inline grid width bypassed the mobile breakpoint, squeezing the meter
   track to 93px. Both fixed; verified at 375×812 with no horizontal overflow.

### Verified, and not verified

Verified in-browser: static mount, example queries, the answered path, the
**refused** path (score gate stops "what is the wifi password" at 0.7964 vs the
0.8167 gate, no generation call spent, 8.35ms server total), extractive-mode
labelling under the 401, citations, all three panels, desktop and mobile, and a
clean console.

**Not verified: the microphone.** The automation browser blocks device capture,
so Web Speech capture and `SpeechSynthesis` playback are written and wired but
have not been exercised end to end. They need a manual pass in Chrome or Edge
over HTTPS/localhost. The typed fallback is a fallback and is labelled as one in
the UI — the task explicitly requires real voice input, so this is the one
remaining thing to confirm by hand before submitting.

### Scope note

§11's cut list says "UI polish beyond functional". That was deliberately
overridden on request: the design now tracks hhgoa.com closely and the panels
are animated. The line held is that every interactive element renders real
pipeline data — none of it is ornament.

### 14.3 End-to-end voice confirmed — and three bugs it exposed

**2026-08-22: a full spoken round trip works in Google Chrome.** Badge reads
`voice ready / running in Google Chrome`; spoken question → transcript →
retrieval in 7.66ms → cited answer → spoken reply. The voice requirement is met.

The first real voice run immediately exposed three defects that no amount of
typed testing would have found.

#### 1. The STT number was measuring the user, not the system

The panel reported `speech-to-text 5703.70ms`. That is not a system cost — it
timed from `recog.start()`, so it included every second of dead air before
speaking plus the whole utterance. On the one panel whose entire purpose is an
honest latency breakdown, it overstated the pipeline by roughly an order of
magnitude, in the direction that makes the system look worse *and* the number
meaningless.

`onspeechstart` / `onspeechend` make the real decomposition available, so it is
now split three ways and labelled:

```
you speaking (not system time)          5100.00   ← user, hatched + dimmed
speech recognition (after you stop)      420.00   ← the recognizer's real cost
… system stages …
mic press → answer (wall clock)         5703.70   ← what the user actually feels
```

User-time rows are visually demoted and **excluded from the bar scale** — left
in, they squashed every real stage into nothing.

#### 2. Citations looked mis-sorted

Displayed cosines ran `0.8432, 0.8164, 0.8326, 0.8290, 0.8200` — not descending,
which reads as a sorting bug. It is not: under hybrid retrieval **RRF sets the
order while the score shown is the unmodified dense cosine** (§4.7). Rows are
now numbered, and when the sequence is non-monotonic the list says so outright.

Note what was *not* done: silently re-sorting by cosine. That would have made
the display tidy and the ranking a lie — it would no longer show the order the
answer was actually built from.

#### 3. The latency bars had never rendered, once, ever

Every bar measured **0×0 pixels**. `.bar` is a `<span>` inside `.track`, so it
was an inline element, and inline elements ignore `width` and `height`.
`.track` escaped the same fate only because it is a grid item, which CSS
blockifies. One missing `display: block`.

This survived several rounds of "verified the latency panel" because I checked
the **numbers**, which were always correct, and read the bars off low-resolution
screenshots where an empty track and a filled bar look alike. It took querying
`getBoundingClientRect().width` to see it. Lesson: for anything visual, assert on
measured geometry, not on a screenshot glance.

Bars are also now scaled against the largest individual **stage**, excluding the
`sum` rows, which are aggregates of the others and double-counted. Retrieval
still renders as a 1.6% sliver against generation's full-width bar — that is the
true shape of the pipeline and exactly what the panel is for.

---

## 15. The vocabulary gate — a POSITIVE separation result

Found by a real voice test on 2026-08-22: someone spoke romanised Hindi into the
mic, and the transcript sailed through scope and score and got a **cited answer**.

### The measurement

`bge-small-en` is English-only, and `recog.lang = "en-IN"` transcribes Hindi
phonetically into Latin script — so romanised Hindi is a realistic input at an
India-based hackathon, not a contrived one.

```
query                              cosine    verdict
asdf qwer zxcv hjkl                0.8317    ANSWERED   <- keyboard mash
tu pata hai na matlab ...          0.8377    ANSWERED
mujhe registration ke bare mein    0.8458    ANSWERED
kitne log aa rahe hain goa mein    0.8316    ANSWERED
पंजीकरण कब शुरू होता है              0.8044    refused    <- a VALID question
```

Keyboard mash outscored several genuine questions and was answered with a
citation, while a legitimate Devanagari question was refused. This is §4.2's
lesson again: **an embedding always lands somewhere**, so cosine cannot tell a
stranger from a question.

### The fix, and why it is not another threshold

Lexical overlap sees what cosine cannot, and it separates **cleanly**:

```
                        n    min maxBM25    min term-hits
eval: answerable       37         4.8865                2
eval: must-refuse      11         2.8131                3
nonsense / non-English  7         0.0000                0
```

Every one of the 48 labelled queries shares at least 2 terms with the corpus.
Every nonsense query shares exactly **0**. So the gate is a **binary structural
test** — "does this query contain a single word the corpus has ever used?" —
not a tuned cutoff. There is no number to overfit and no distribution to
overlap, which is precisely why it succeeds where §4.2 failed.

Contrast worth keeping: §4.2 and §4.3 are negative results about *similarity*
signals. This is a positive result from a *structural* one. The lesson is not
"try harder thresholds", it is "find a signal with a real discontinuity".

Verified: **0 of 48** eval queries rejected; nonsense and non-Latin input all
stopped at the gate; 19/19 on the guardrail suite.

### Honest limits

- **Necessary, not sufficient.** `"goa asdf qwer"` has one corpus term and
  passes. It catches total strangers, not partial ones.
- **It refuses valid non-English questions.** That is a real product limitation,
  not a fix. But refusing *while saying the source is English-only*
  (`NO_VOCAB_TEXT`) beats answering a Hindi question with a confidently wrong
  English passage, which is what happened before.
- Reused machinery: BM25 already existed for §4.7 hybrid retrieval, so the gate
  cost no new dependency and no new artifact. BM25 is now built in **both**
  retrieval modes, since the gate needs it even when fusion is off.

### A bug this nearly shipped with

The first implementation guarded with `if n_terms and term_hits == 0`. The
tokenizer matches `[a-z0-9]+`, so a Devanagari query yields **zero** terms,
`n_terms` is falsy, and the query was waved through. It *looked* correct only
because that one test query also happened to fall below the score gate — a
non-Latin query scoring above it would have been answered. Empty input is
already handled by `check_scope`, so the condition is now simply
`term_hits == 0`. Case g043 pins the **gate that fires**, not just the outcome,
so the coincidence cannot come back.
