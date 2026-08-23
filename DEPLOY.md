# Deploying Ask HH Goa

Two pieces, and they do **not** both go to the same place.

| piece | what it is | where it can run |
|---|---|---|
| frontend | `app/static/index.html`, one self-contained file | Netlify, or served by the API itself |
| API | FastAPI + ONNX Runtime + FAISS + a ~130MB model | anywhere that runs Python — **not Netlify** |

---

## Why the API cannot go on Netlify

Worth stating plainly so nobody burns an afternoon finding out at deploy time:

- **Netlify Functions run JavaScript/TypeScript and Go.** This backend is
  Python. There is no Python runtime to target.
- Even ignoring language: the bundle would need `onnxruntime`, `faiss-cpu`,
  `numpy` and the embedding weights — far past the function size limit.
- Functions are serverless, so every cold start reloads the model. This project
  is *measured on latency*; a multi-second cold start on the critical path
  would invalidate the one claim it makes.

So Netlify hosts the static page, and the page is told where the API lives.

---

## Option 0 — no backend at all (lite mode)

A static host with **no API** still works. The page loads a 37KB
`corpus.json` and runs retrieval in the browser.

Nothing to configure: if the startup probe finds no API, lite mode activates
automatically and the banner says so.

**What you give up, measured on the same 37-query eval set:**

| | recall@5 | prec@1 | MRR |
|---|---|---|---|
| full hybrid (API) | 0.919 | 0.703 | 0.763 |
| **lite (browser)** | **0.865** | **0.649** | **0.733** |

Lite mode is **BM25 only** — a browser has no embedding model, so there are no
dense vectors and no RRF fusion. The score gate is also skipped, because its
threshold is a calibrated *cosine* and BM25 scores are on an unrelated scale.
Scope filtering, the vocabulary gate, parent collapsing and citations are all
unchanged.

It exists so a frontend-only deploy degrades instead of dying. Use a real API
for anything you want judged on latency or ranking quality.

Regenerate the corpus after any chunking change:

```bash
py -m app.chunking.export_static
```

## Option A — everything on one host (simplest)

The API already serves the frontend at `/`. Nothing to configure.

```bash
py -m uvicorn app.main:app --port 7860
```

**Hugging Face Spaces (Docker, free CPU)** is the intended target: the
Dockerfile listens on 7860, runs as uid 1000, and bakes the model into the
image so startup does not pay a download.

1. Create a Space → SDK **Docker**.
2. Push this repo to it.
3. Optionally set `GEMINI_API_KEY` in Space **Settings → Variables and secrets**.
   Leave it unset and the Space runs in extractive mode, which is fully
   functional — see "Running without a key" below.

## Option B — frontend on Netlify, API elsewhere

1. Deploy the API (Option A) and note its public URL, e.g.
   `https://<user>-ask-hh-goa.hf.space`.
2. Connect this repo to Netlify. `netlify.toml` already sets
   `publish = "app/static"` and the build command.
3. In Netlify → **Site settings → Environment variables**, set:

   ```
   API_BASE = https://<user>-ask-hh-goa.hf.space
   ```

   `netlify/inject-api-base.sh` bakes that into the page at build time. Leave it
   unset and the page falls back to same-origin, then to whatever the visitor
   enters in the Setup panel.
4. On the API side, restrict CORS to your Netlify origin:

   ```
   ALLOWED_ORIGINS = https://your-site.netlify.app
   ```

   It defaults to `*`, which is fine for a public read-only demo but worth
   narrowing once the URL is stable.

### Pointing the page at an API without redeploying

```
https://your-site.netlify.app/?api=https://your-api.hf.space
```

Or type it into the Setup panel — it persists in `localStorage`.

---

## Running without a key

**The API key is optional, and the app is genuinely useful without one.**

| | no key | with a key |
|---|---|---|
| voice input | ✅ | ✅ |
| retrieval, hybrid ranking | ✅ | ✅ |
| citations with scores | ✅ | ✅ |
| all four guardrails | ✅ | ✅ |
| latency instrumentation | ✅ | ✅ |
| answer text | the retrieved passage **verbatim** | written prose, grounded and cited |

With no key the service answers **extractively**: it returns the best retrieved
passage as-is. That is the same code path it degrades to when the free-tier
quota runs out, so the degraded mode is continuously exercised rather than
being untested emergency code — and CI can integration-test the whole retrieval
path with no secrets at all.

Three ways to supply a key, in precedence order:

1. **Setup panel in the UI** (bring-your-own-key). Stored in `localStorage`, or
   `sessionStorage` via **Session only** on a shared machine. Sent per request
   as an `X-Gemini-Key` header — never a query parameter, because a key in a
   URL leaks into browser history, server logs and `Referer`. Never logged and
   never persisted server-side.
2. **`GEMINI_API_KEY`** in the server environment — used when the request
   carries no key of its own.
3. Neither → extractive mode.

Get a key free at <https://aistudio.google.com/apikey>. It is **39 characters
and starts with `AIza`**. A Vertex AI or service-account credential is a
different thing entirely and will 401 on every call — the UI detects the wrong
shape and says so.

---

## Environment variables

| variable | default | purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | optional; enables generated answers |
| `ALLOWED_ORIGINS` | `*` | comma-separated CORS origins for a split deploy |
| `API_BASE` | *(unset)* | **Netlify build only** — bakes the API URL into the page |
| `EMBED_MODEL` | `BAAI/bge-small-en` | **not** the `-v1.5` variant; that ships int8-quantized and measures 23x slower |
| `RETRIEVAL_MODE` | `hybrid` | `dense` reverts to embedding-only ranking |
| `RETRIEVAL_MIN_SCORE` | `0.8167` | calibrated, not guessed — see `bench/calibrate.py` |

Full annotated list in `.env.example`.

---

## Verifying a deploy

```bash
curl -s https://<your-api>/healthz
# {"status":"ok","vectors":91,"generation_enabled":false,"accepts_user_key":true}
```

Then load the frontend and ask **"what is the wifi password"** — it should
**refuse** at the score gate, showing the pipeline stopping. A deploy that
answers that question has a broken guardrail chain.

Note the voice input needs **Google Chrome or Edge** over HTTPS. Chrome's Web
Speech API is not on-device — it uploads audio to Google and only builds
carrying Google's API key are accepted, so Brave, plain Chromium and in-app
browsers fail with a `network` error. The page detects this and says so; the
typed fallback works everywhere.
