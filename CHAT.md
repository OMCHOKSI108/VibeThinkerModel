# VibeThinker Local Chat

A local chatbot / coding-helpbot UI for **VibeThinker-3B**, running entirely on
your own machine. FastAPI backend, Hugging Face Transformers for inference,
and a static HTML/CSS/JS frontend — no React, no build step, no mock
responses.

## What this is

- A single FastAPI server that loads `OMCHOKSI108/VibeThinker-3B` once at
  startup and keeps it resident in memory.
- A `/api/chat/stream` endpoint that streams generated tokens to the browser
  over Server-Sent Events (SSE) as the model produces them.
- A small GraphQL API (`/graphql`) for status/metadata queries and session
  reset — things that are simple request/response operations.
- A static frontend (`client/`) that renders a ChatGPT-style dark UI, with
  Markdown formatting, code blocks with copy buttons, and a "thinking" bar
  while the model is working.

## Why both GraphQL and SSE

GraphQL is a good fit for typed, one-shot operations: `health`, `modelInfo`,
and the `resetSession` mutation all return a fixed shape and don't need a
stream. Token-by-token chat output is a different shape of problem — a long
sequence of incremental text — and GraphQL subscriptions would require extra
transport machinery (websockets or multipart responses) to do what plain HTTP
SSE already does cleanly. So: GraphQL for metadata/control, SSE for the actual
token stream.

## Project layout

```
local-vibethinker-chatbot/
├── server/            FastAPI + GraphQL + SSE backend
│   └── app/
│       ├── main.py             FastAPI app, lifespan model load, routes/mounts
│       ├── config.py           Env var loading
│       ├── model_loader.py     Loads tokenizer/model once (CUDA/4-bit aware)
│       ├── intent_detector.py  Keyword-pattern intent classification (debug/explain/build/etc.)
│       ├── memory_store.py     Per-session conversation memory: recent turns + folded summary
│       ├── context_manager.py Builds the full message list (prompts + memory + current turn)
│       ├── prompts/            system_prompt.txt (identity/honesty/safety), developer_prompt.txt (project rules)
│       ├── generation.py       model.generate() in a thread + smart SSE buffering
│       ├── graphql_schema.py   Strawberry schema: health, modelInfo, resetSession
│       ├── sse_routes.py       POST /api/chat/stream
│       └── utils.py            SSE formatting, think-tag stripping, stop signal
└── client/            Static HTML/CSS/JS frontend (served at /client)
```

## How conversation memory works

Each chat turn goes through `context_manager.build_messages`, which assembles,
in order: the system prompt, the developer prompt, an intent-specific
formatting hint (from `intent_detector.detect_intent`), a folded summary of
older turns (if any), the recent raw turns, and the current message.
`memory_store.py` holds this per `session_id` entirely in process memory — no
database, no disk, no network call. Once a session's history passes
`MAX_RECENT_MESSAGES` (12), the oldest turn is folded into a short truncated
summary line instead of being kept verbatim, so context survives indefinitely
without an unbounded transcript. The `resetSession` GraphQL mutation clears a
session's memory.

## About the model

- `OMCHOKSI108/VibeThinker-3B` is the Hugging Face mirror used for weights.
  Transformers downloads and caches the weights the first time you run the
  server — they are **not** stored in this repository.
- The [VibeThinkerModel GitHub repo](https://github.com/OMCHOKSI108/VibeThinkerModel)
  contains docs, evals, and notebooks for context on the model — it does **not**
  contain the weights themselves. Clone it only if you want background reading:
  ```bash
  git clone https://github.com/OMCHOKSI108/VibeThinkerModel
  ```
- The server loads the model with `AutoModelForCausalLM` /
  `AutoTokenizer` from `transformers`. If you already have local weights, point
  `MODEL_PATH` in `.env` at that directory and it takes precedence over
  `MODEL_ID`.

## How model loading works

`server/app/model_loader.py` runs once, from the FastAPI `lifespan` hook in
`main.py`, before the app starts serving requests:

1. Picks the device: CUDA if available, else CPU.
2. If `LOAD_IN_4BIT=true`, builds a `BitsAndBytesConfig` (requires CUDA and
   `bitsandbytes`) and loads with `device_map="auto"`.
3. Otherwise loads in `float16` on CUDA (`device_map="auto"`) or `float32` on
   CPU.
4. On any failure (missing bitsandbytes, OOM, bad model id, etc.) the error is
   captured in `model_loader.state.load_error` — the server still starts, but
   `/health` and the chat endpoint report the model as unavailable instead of
   faking a response.

## Setup

```bash
cd local-vibethinker-chatbot/server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
# edit ../.env if needed (model id/path, 4-bit, port, etc.)
python run.py
```

Then open:

```
http://127.0.0.1:8000/client
```

The frontend is served directly by FastAPI from `client/`, so there's no
separate frontend server or build step.

## Example requests

GraphQL health query:

```bash
curl -s http://127.0.0.1:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"query { health { status modelLoaded device } }"}'
```

SSE chat stream:

```bash
curl -N -s http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"Write a Python function that reverses a linked list."}'
```

## Low VRAM notes (e.g. RTX 3050 4GB)

- Set `LOAD_IN_4BIT=true` in `.env` (requires `bitsandbytes` and a CUDA GPU).
- Lower `MAX_NEW_TOKENS` (e.g. 256–512) to reduce peak memory and latency.
- Close other GPU-using applications (browsers with hardware acceleration,
  other model servers, etc.) before starting the server.
- If you still hit CUDA OOM mid-generation, the server returns a clear SSE
  `error` event instead of crashing — retry with a shorter prompt or fewer
  max tokens.

## Troubleshooting

- **`/health` shows `model_loaded: false`** — check the server logs for the
  captured `load_error`. Common causes: wrong `MODEL_ID`, no internet access
  to Hugging Face on first run, or `LOAD_IN_4BIT=true` without a CUDA device.
- **`LOAD_IN_4BIT` fails to load** — confirm `bitsandbytes` is installed and
  you're on a CUDA-capable GPU; 4-bit loading is not supported on CPU.
- **Chat request hangs at "Thinking..."** — the model may be generating a long
  `<think>` block before producing visible output; this is expected behavior,
  not a bug — the UI intentionally hides chain-of-thought tokens.
- **Stop button doesn't seem to stop fast** — generation halts at the next
  decode step after the abort is detected, not mid-token; for very large
  `max_new_tokens` this can take a moment.
- **CORS errors opening `index.html` directly via `file://`** — always use
  `http://127.0.0.1:8000/client`, not the raw file path, so requests share the
  server's origin.
