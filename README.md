# 🎥 Multilingual Video RAG Chatbot

Chat with any YouTube video in your own language. Paste a link, and the app downloads the video's transcript, indexes it, and answers your questions using **only what the video actually says**. Every answer is checked against the transcript before it is shown, and it comes with clickable timestamps that jump to the moments it is based on.

Built with **Gradio**, **LangChain**, **FAISS + BM25 hybrid search**, a **cross-encoder reranker** and an LLM from **Hugging Face** or **Ollama** (local).

---

## Table of contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Repository layout](#repository-layout)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration (.env)](#configuration-env)
7. [Running the app](#running-the-app)
8. [Using the app](#using-the-app)
9. [Choosing a language model](#choosing-a-language-model)
10. [Saving conversations with PostgreSQL](#saving-conversations-with-postgresql)
11. [Answer checking (grounding)](#answer-checking-grounding)
12. [Multilingual support](#multilingual-support)
13. [Running with Docker](#running-with-docker)
14. [Testing](#testing)
15. [Logging and troubleshooting](#logging-and-troubleshooting)
16. [Security and privacy](#security-and-privacy)
17. [Limits and performance](#limits-and-performance)
18. [Known limitations](#known-limitations)
19. [Contributing](#contributing)
20. [Author and license](#author-and-license)

---

## Features

| Feature | What it does |
| --- | --- |
| **Ask questions about any video** | Works with `youtube.com/watch`, `youtu.be`, `/shorts/`, `/embed/`, `m.youtube.com` and `music.youtube.com` links. |
| **Answers grounded in the transcript** | The model must quote the transcript; each sentence of the answer is checked against those quotes. Unsupported answers are replaced with a verified quote or an honest "not enough information". |
| **Timestamp citations** | Up to three clickable `mm:ss` links per answer, pointing to the exact moments in the video. Only quotes that passed the check get links. |
| **Follow-up questions** | "Why?" or "What about the second one?" are rewritten into full questions using the conversation so far. |
| **Multilingual** | Answers come back in the language you ask in. The app's own messages are translated into Hindi, Spanish, French, German, Portuguese and Italian. |
| **Video details** | Questions such as "Which channel uploaded this?", "How long is it?" or "How many views?" are answered straight from the video's details, with no model call. |
| **Several conversations** | The sidebar keeps one conversation per video. Switch between them at any time. |
| **Transcript viewer** | Shows the full timestamped transcript. |
| **Retrieved-context viewer** | Shows exactly which transcript chunks the last answer was based on, and how far the request got. |
| **Suggested questions** | One-click starters: main concepts, simple explanation, examples, key takeaways. |
| **Export notes** | Downloads the conversation, with its source references, as a Markdown file. |
| **Optional persistence** | With PostgreSQL configured, conversations survive page refreshes and server restarts. |
| **Light and dark themes** | Follows your system theme and adapts to small screens and phones. |
| **Usage limits** | Per-browser hourly limits and size limits protect your API credits and the server. |

---

## How it works

```mermaid
flowchart LR
    A[YouTube link] --> B[Video details<br/>yt-dlp]
    A --> C[Transcript<br/>youtube-transcript-api]
    C --> D[Split into 800-char chunks<br/>120 overlap]
    D --> E[FAISS index<br/>all-MiniLM-L6-v2]
    D --> F[BM25 index]
    Q[Your question] --> R{About the video<br/>itself?}
    R -- yes --> M[Answer from video details]
    R -- no --> W[Rewrite follow-up<br/>into a full question]
    W --> H[Hybrid search<br/>40% BM25 + 60% FAISS]
    E --> H
    F --> H
    H --> K[Rerank with<br/>bge-reranker-base, keep top 4]
    K --> L[LLM writes answer<br/>+ exact quotes, as JSON]
    L --> V[Check every sentence<br/>against the quotes]
    V --> T[Answer + timestamp links]
```

Step by step:

1. **Load the video** (`youtube_source.py`)
   - The link is validated, and the 11-character video id is extracted.
   - yt-dlp reads the title, channel, description, upload date, duration and view count.
   - youtube-transcript-api lists the caption tracks and picks one. English is preferred, then Hindi, then any other language (regional codes such as `en-US` count as English). Captions written by people are preferred over auto-generated ones.
   - The transcript's language is detected with `langdetect`.
2. **Index it** (`build_hybrid_retriever` in `app_yt.py`)
   - The transcript is split into 800-character chunks with 120 characters of overlap.
   - Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` into a **FAISS** index (search by meaning).
   - The same chunks go into a **BM25** index (search by exact words).
   - Both are combined in an ensemble retriever: 40% BM25, 60% FAISS, 10 results each.
3. **Route the question**
   - A set of patterns catches questions about the video itself (channel, title, description, duration, views). These are answered from the video details without calling the model.
   - Content questions that merely mention a word such as "views" or "title" still go to the transcript.
4. **Rewrite follow-ups**
   - From the second question on, the model rewrites the question into a standalone one using the last 8 messages.
   - The first question skips this step, which saves a model call.
5. **Retrieve and rerank**
   - The hybrid retriever returns up to 20 candidate chunks.
   - The cross-encoder `BAAI/bge-reranker-base` scores each chunk against the question, and the best 4 are kept.
6. **Generate**
   - The model receives the 4 chunks, the conversation history, the video's title and channel, and the language to answer in.
   - It must return JSON with an `answer` and one or two short **exact quotes** from the transcript as `evidence`.
7. **Check the answer** (`grounding.py`)
   - Every quote must appear word for word in the retrieved text.
   - Numbers in the answer must appear in the quotes.
   - Each sentence is checked by a verifier model, in parallel.
   - If the check fails, the app tries once to find a single quote that answers the question, and rewords it if that rewording also passes the check. Otherwise it answers that the video does not contain enough information.
8. **Cite**
   - Each verified quote is located in the timed transcript, and links such as `https://www.youtube.com/watch?v=ID&t=125s` are added under the answer.

---

## Repository layout

```
Multilingual_Video_RAG_Chatbot/
├── README.md                       ← this file
├── .gitignore
└── youtube-chatbot-main/           ← the application
    ├── app_yt.py                   settings, models, retrieval, prompts, sessions, Gradio UI
    ├── youtube_source.py           link parsing, video details, transcripts, YouTube errors
    ├── grounding.py                answer checking, citations, translated fixed replies
    ├── grounding_ensemble.py       optional local verifier (HHEM + NLI)
    ├── factual_support.py          local NLI classifier used by the verifier
    ├── chat_store.py               PostgreSQL storage
    ├── app_logging.py              log format with a short id per request
    ├── inference_errors.py         readable model error messages, never leaking tokens
    ├── system_memory.py            RAM checks on Windows, Linux and macOS
    ├── app_style.css               page styling (light and dark)
    ├── download_models.py          fetches the optional local verifier weights
    ├── check_inference.py          tests the Hugging Face connection
    ├── evaluate_factual_support.py benchmark for the local classifier
    ├── inspect_nli_environment.py  shows RAM and whether verifier weights are present
    ├── download_nli_model.py       older single-model downloader (superseded)
    ├── benchmarks/                 benchmark cases
    ├── local_models/hhem-2.1-open/ reviewed HHEM code, pinned by SHA-256 (weights not in git)
    ├── tests/                      automated test suite (pytest)
    ├── scripts/                    older manual end-to-end runners
    ├── .github/workflows/tests.yml CI: runs the tests on every push
    ├── .env.example                every setting, with defaults
    ├── requirements.txt            pinned runtime dependencies
    ├── requirements-ollama.txt     extra package for the Ollama provider
    ├── requirements-dev.txt        runtime + pytest
    ├── pyproject.toml              pytest and black settings
    ├── Dockerfile
    ├── README.md                   shorter project README
    ├── GROUNDING_ENSEMBLE.md       notes on the experimental local verifier
    └── OLLAMA_VERIFICATION.md      notes on testing with Ollama
```

---

## Requirements

- **Python 3.11.** The dependencies are pinned and tested on 3.11.
- **About 2 GB of free disk space** for Python packages and the embedding and reranker models, which download on first start.
- **At least 4 GB of RAM.** More is needed if you use Ollama or the local verifier.
- One of these language models:
  - a **Hugging Face token** with the "Make calls to Inference Providers" permission, plus inference credits on the account, **or**
  - **Ollama** running locally.
- Optional: **PostgreSQL 13+**, so conversations survive restarts.
- Optional: **Docker**.

---

## Installation

```bash
git clone https://github.com/akshaya3790/Multilingual_Video_RAG_Chatbot.git
cd Multilingual_Video_RAG_Chatbot/youtube-chatbot-main

python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell or cmd)
source .venv/bin/activate         # Linux / macOS

pip install -r requirements.txt
```

> **Tip (CPU-only machines):** install the CPU build of PyTorch first. It is much smaller than the default CUDA build:
> `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`

Then create your settings file:

```bash
copy .env.example .env            # Windows
cp .env.example .env              # Linux / macOS
```

Open `.env` and set at least `HF_TOKEN` (or `LLM_PROVIDER=ollama`, see below). Never commit `.env`; it is already in `.gitignore`.

---

## Configuration (.env)

Every setting is optional except the model credentials. `.env.example` lists them all with their defaults. Restart the app after changing `.env`.

### Answer model

| Setting | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | *(required for Hugging Face)* | Hugging Face access token. |
| `LLM_PROVIDER` | `huggingface` | `huggingface` or `ollama`. There is no automatic fallback between them. |
| `HF_LLM_MODEL` | `openai/gpt-oss-20b` | Hugging Face model that writes the answers. |
| `HF_VERIFIER_MODEL` | *(unset)* | A second Hugging Face model that checks the answers. Unset, the answer model checks itself. |
| `LLM_MAX_TOKENS` | `1536` | Maximum length of a model reply. Reasoning models spend part of this on thinking; raise it if the log says an answer was cut off. |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Ollama model, used when `LLM_PROVIDER=ollama`. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Address of the Ollama server. |
| `OLLAMA_NUM_CTX` | `8192` | Ollama context window, in tokens. |
| `OLLAMA_VERIFIER_MODEL` | *(unset)* | A second Ollama model that checks the answers. |
| `GROUNDING_CHECKER` | `llm` | `llm`, or `local_ensemble` for the experimental local verifier. |

### Saving conversations

| Setting | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | *(unset)* | PostgreSQL connection string. Without it, conversations live in memory only. |
| `CHAT_RETENTION_DAYS` | `30` | Conversations untouched for this long are deleted on startup. |
| `BROWSER_STATE_SECRET` | *(unset)* | Key that encrypts the browser's saved session id. If unset and a database is configured, one is generated and kept in the database. |

### YouTube

| Setting | Default | Description |
| --- | --- | --- |
| `YOUTUBE_PROXY` | *(unset)* | HTTP(S) proxy for YouTube, for servers that YouTube blocks (common on cloud hosting). |

### Limits

| Setting | Default | Description |
| --- | --- | --- |
| `MAX_QUESTION_CHARS` | `1000` | Longest question accepted. |
| `MAX_VIDEO_MINUTES` | `180` | Longest video accepted. |
| `MAX_TRANSCRIPT_CHARS` | `250000` | Longest transcript that will be indexed. |
| `MAX_VIDEOS_PER_HOUR` | `20` | Video loads per browser per hour. |
| `MAX_QUESTIONS_PER_HOUR` | `120` | Questions per browser per hour. |
| `SESSION_IDLE_MINUTES` | `120` | Idle sessions are removed from memory after this long. |
| `MAX_SESSIONS` | `50` | Sessions kept in memory; the least recently used are dropped first. |
| `MAX_CHATS_PER_SESSION` | `10` | Conversations per browser; the oldest inactive ones are dropped first. |
| `MAX_CONCURRENT_REQUESTS` | `4` | Requests processed at the same time. |

### Logging

| Setting | Default | Description |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |

---

## Running the app

```bash
cd youtube-chatbot-main
python app_yt.py
```

Open **http://127.0.0.1:7860** in your browser.

- On the first start, the embedder (~90 MB) and reranker (~1 GB) download from Hugging Face. They are cached afterwards. They load in the background while the page opens, so the first video may take a little longer.
- To reach the app from other devices on your network, set `GRADIO_SERVER_NAME=0.0.0.0` (and optionally `GRADIO_SERVER_PORT`) before starting it.
- Stop the app with **Ctrl+C**.

---

## Using the app

1. **Paste a YouTube link** into the URL box and click **Load Video**. The status card shows the title, channel, transcript language and duration once the video is indexed. This takes a few seconds for most videos.
2. **Ask a question** in the message box and press **Enter** or click **Send**. You can also click one of the suggested questions.
3. **Follow the timestamps** under an answer to jump to the matching moment in the video.
4. **Ask follow-ups** naturally, for example "Why does he say that?" or "Give me an example".
5. **Ask about the video itself**: "Who uploaded this?", "What is the title?", "How long is the video?", "How many views does it have?", "What does the description say?"
6. **Inspect** with **View Transcript** (the full transcript with timestamps) and **View Retrieved Context** (the chunks used for the last answer, the request stage, and any error stage).
7. **Manage conversations** in the sidebar:
   - **+ New Chat** starts a new conversation.
   - Clicking a previous video switches to its conversation.
   - **Clear Current Chat** empties the current conversation.
   - **Export Notes (.md)** downloads it.

Only one answer is generated at a time per browser. A second click while an answer is being generated shows a short "still answering" notice.

---

## Choosing a language model

### Hugging Face (default)

1. Create a token at https://huggingface.co/settings/tokens with the **"Make calls to Inference Providers"** permission.
2. Make sure the account has inference credits (**Settings → Billing**).
3. Put the token in `.env` as `HF_TOKEN=...`.
4. Optionally test the connection without printing the token:

   ```bash
   python check_inference.py
   ```

Any chat model available through Hugging Face Inference Providers can be set as `HF_LLM_MODEL`. For an independent check, set `HF_VERIFIER_MODEL` to a different model.

### Ollama (local, free, private)

1. Install Ollama from https://ollama.com/download.
2. Pull a model and install the extra package:

   ```bash
   ollama pull qwen2.5:1.5b
   pip install -r requirements-ollama.txt
   ```

3. Set `LLM_PROVIDER=ollama` in `.env`. `HF_TOKEN` is not needed.

With Ollama, the app asks the model for structured JSON output. Small models are fast but often answer "not enough information"; a 7B model (for example `qwen2.5:7b`) gives much better answers if your machine can run it.

---

## Saving conversations with PostgreSQL

Without a database, conversations disappear when the app restarts. To keep them:

```sql
-- in psql, as the postgres user
CREATE USER ytchat WITH PASSWORD 'your_password';
CREATE DATABASE ytchat OWNER ytchat;
```

```env
DATABASE_URL=postgresql://ytchat:your_password@localhost:5432/ytchat
```

On startup, the app:
- creates its tables (`chats` and `app_settings`);
- deletes conversations older than `CHAT_RETENTION_DAYS`;
- generates a browser key if needed.

What is stored:
- **Saved:** video details, transcript, and messages with their source references.
- **Not saved:** search indexes, which are rebuilt from the transcript the first time you ask a question in a restored conversation (it takes a few seconds).

Each browser keeps an encrypted session id, so after a refresh or restart its conversations reappear. If the database cannot be reached, the app still runs and logs a warning; conversations then live in memory only.

---

## Answer checking (grounding)

The checker is the reason answers stay faithful to the video. It runs after the model answers and before anything is shown.

| Check | Rejects |
| --- | --- |
| Valid JSON with `answer` and `evidence` | Malformed or truncated model output |
| Each quote appears verbatim in the retrieved text | Invented or paraphrased "quotes" |
| Every number in the answer appears in the quotes | Made-up figures and dates |
| At most 3 quotes and at most 6 sentences | Rambling, stitched answers |
| No links or HTML written by the model | Links the model added (only the app's own timestamp links are shown) |
| Each sentence is supported **and** answers the question (verifier model, run in parallel) | Unsupported claims and off-topic summaries |

When the check fails, the app does not just give up:

1. It asks the model to pick the single quotation, among the retrieved text, that best answers the question.
2. It verifies that quotation.
3. It asks the model to reword it as a direct answer, and keeps the rewording only if that also passes the check.
4. If the rewording fails, it shows the verified quote itself: *The video says: "…"*.
5. If nothing passes, the answer is: *"The available video context does not provide enough information to answer this question."*

### Optional local verifier

`GROUNDING_CHECKER=local_ensemble` replaces the model-based check with two local classifiers: **HHEM-2.1-Open** (Vectara) and **NLI DeBERTa-v3-xsmall**.

- **Setup:** fetch the pinned weights (about 700 MB, verified by hash) with `python download_models.py`.
- **Tamper protection:** the small HHEM code files in `local_models/hhem-2.1-open/` are pinned by SHA-256, and the app refuses to load them if they change.
- **Status:** experimental and English-only. It needs at least 1.5 GB of free RAM. See `GROUNDING_ENSEMBLE.md` for details and measurements.

---

## Multilingual support

- **Transcripts:** any language YouTube provides. English is preferred, then Hindi, then any other.
- **Questions:**
  - Answers come back in the language of your question.
  - Short questions such as "why?" or "क्यों?" are judged by their alphabet, because statistical detection is unreliable on a few letters.
  - Language detection is seeded, so the same question always gets the same language.
- **Fixed replies:** "not enough information", "could not verify" and the error and quote messages are translated into **Hindi, Spanish, French, German, Portuguese and Italian**. Other languages fall back to English for these messages only.
- **Sentence splitting:** answers are split into sentences before checking. The splitter handles abbreviations ("Dr.", "e.g.", "U.S.") and the Hindi full stop "।".

---

## Running with Docker

```bash
cd youtube-chatbot-main
docker build -t youtube-chat .
docker run --rm -p 7860:7860 --env-file .env -v hf-cache:/app/.cache/huggingface youtube-chat
```

- The image is based on `python:3.11-slim` with CPU-only PyTorch, and runs as a non-root user.
- `.env` is passed at run time with `--env-file`, never copied into the image.
- The `hf-cache` volume keeps downloaded models between runs.
- **PostgreSQL on your computer:** use `host.docker.internal` instead of `localhost` in `DATABASE_URL`.
- **Ollama:** build with `--build-arg INSTALL_OLLAMA=true`. Then run with `-e LLM_PROVIDER=ollama -e OLLAMA_BASE_URL=http://host.docker.internal:11434`. On Linux, also add `--add-host=host.docker.internal:host-gateway`.
- **Local verifier:** run `python download_models.py`, then add `-v "$PWD/local_models:/app/local_models:ro"`.

---

## Testing

```bash
cd youtube-chatbot-main
pip install -r requirements-dev.txt
python -m pytest
```

- **180 tests**, run in about 10 seconds.
- They never download or call a real model: `tests/conftest.py` swaps in small fakes for the embedder, reranker and LLM. To use the real models, set `YTCHAT_REAL_MODELS=1`.
- The PostgreSQL tests run only when `TEST_DATABASE_URL` points to a **throwaway** database, because they empty its tables.
- **GitHub Actions** (`.github/workflows/tests.yml`) runs the whole suite on every push and pull request, including PostgreSQL in a service container.

What the tests cover:
- link parsing and YouTube errors;
- caption track choice;
- hybrid retrieval wiring;
- the metadata router;
- language detection;
- every grounding rule and the fallback path;
- citations and timestamps;
- sessions, eviction, rate limits and the one-answer-at-a-time guard;
- PostgreSQL storage;
- the transcript and context viewers;
- export;
- the suggestion handlers;
- the helper scripts and model downloader.

`scripts/` contains older manual end-to-end runners. `run_e2e_acceptance_tests.py` needs live access to YouTube.

---

## Logging and troubleshooting

Each log line shows the time, the level and a short id for the question or video load it belongs to, plus the time spent in each stage:

```
2026-10-27 03:13:53 INFO    [cd0e3045] ytchat.app: question answered in 6.2s, result verified, stages rewriting 0.8s, retrieving 0.1s, reranking 0.4s, generating 3.9s, verifying 1.0s
```

Errors are shown in the chat but **not saved** to the conversation, so they never reach later prompts. The question stays in the box so you can send it again.

| Problem | Cause and fix |
| --- | --- |
| **402 Payment Required** | The Hugging Face account has no inference credits. Add credits under Settings → Billing, or switch to Ollama. |
| **401 / 403** | Invalid token, or one without the "Make calls to Inference Providers" permission. Replace `HF_TOKEN` and restart. |
| **429** | Rate limited by Hugging Face. Wait a moment. |
| **"YouTube blocked this server"** | Common on cloud hosts. Run the app on your own computer or set `YOUTUBE_PROXY`. |
| **"No transcript"** | The video has no captions, so it cannot be used. |
| **"Video unavailable"** | Private, removed, age-restricted or region-locked video. |
| **"Video too long" / "Transcript too long"** | Over the limits. Raise `MAX_VIDEO_MINUTES` or `MAX_TRANSCRIPT_CHARS` if your machine can handle it. |
| **Answers are often "not enough information"** | Try a larger model, or set a separate `HF_VERIFIER_MODEL`. Check **View Retrieved Context** to see whether the right chunks were found. |
| **Log says "answer looks cut off"** | Raise `LLM_MAX_TOKENS`. |
| **Ollama connection errors** | Start Ollama (`ollama serve`), check `OLLAMA_BASE_URL`, and run `ollama list` to confirm that the model is installed. |
| **Conversations vanish after restart** | Set `DATABASE_URL`. Check the startup log for a PostgreSQL warning. |

---

## Security and privacy

- **Secrets** stay in `.env`, which is excluded from git and from the Docker image. Error messages and logs never include tokens, passwords or request contents.
- **Untrusted content:** transcripts, titles and descriptions come from YouTube and are treated as untrusted.
  - They are passed to the model as data, never as instructions.
  - They are shown as plain text, so a video cannot inject links, images or HTML into the page.
- **Links:** links written by the model are removed; the only links shown are the app's own checked timestamp links.
- **Downloads:** exported notes are temporary files, removed after an hour.
- **Local verifier:** its code is hash-pinned and its weights are hash-checked.
- **Sessions:** browser session ids are random UUIDs, stored encrypted in the browser.

---

## Limits and performance

- **First answer of a session:** 3–10 seconds with a hosted model, depending on the provider. Ollama on CPU can take much longer.
- **Model calls per question:**
  - one call to rewrite the question (from the second question on);
  - one call to generate the answer;
  - one checking call per sentence, run in parallel;
  - one to three more calls if the fallback runs.
- **Memory:** every loaded video keeps a FAISS index, a BM25 index and its transcript in memory. The session and conversation limits above bound the total.

---

## Known limitations

- Answer checking uses a model, so it can still be wrong, and small models often refuse questions the video does answer.
- Videos without captions cannot be used. Speech-to-text is not included.
- Search indexes live in memory in a single app process, so the app is not built to scale horizontally.
- Transcript availability and video details depend on YouTube, which can block or change its services.
- The local verifier is experimental and English-only.

---

## Contributing

1. Fork the repository and create a branch.
2. Install the development requirements: `pip install -r youtube-chatbot-main/requirements-dev.txt`.
3. Make your change, and run `python -m pytest` inside `youtube-chatbot-main/`.
4. Format with black: `black -l 100 -S app_yt.py`.
5. Open a pull request. CI runs the tests automatically.

Never commit `.env`, model weights, or files from `project_snapshots/` or `verification_logs/`.

---

## Author and license

Maintained by [akshaya3790](https://github.com/akshaya3790).

Released under the **MIT License**.
