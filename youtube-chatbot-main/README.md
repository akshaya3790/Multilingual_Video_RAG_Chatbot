# 🎥 YouTube RAG Chat

Chat with any YouTube video. Paste a link, and the app fetches the transcript, indexes it, and answers your questions using only what the video says, with timestamp links to the parts it used.

## How it works

1. **Load a video.** `youtube_source.py` gets the title, channel and other details (yt-dlp) and the transcript (youtube-transcript-api). English captions are preferred, then Hindi, then any other; captions written by people beat auto-generated ones.
2. **Index it.** The transcript is split into 800-character chunks (120 overlap) and indexed twice: FAISS with `all-MiniLM-L6-v2` embeddings for meaning, and BM25 for exact words. The two are combined (40% BM25, 60% FAISS).
3. **Route the question.** Questions about the video itself ("which channel uploaded this?", "how long is it?") are answered from its details, without a model call. Everything else goes to the transcript.
4. **Retrieve.** Follow-up questions are rewritten into standalone ones using the conversation so far. The best matches from both indexes (up to 20 chunks) are reranked by a cross-encoder (`bge-reranker-base`), and the top 4 go to the model.
5. **Answer and check.** The model must answer from those chunks and quote its evidence. Each sentence is then checked against the quotes; unsupported answers are replaced by a verified quote or by "not enough information". Only quotes that passed the check get timestamp links.

Answers follow the language of your question (short questions such as "why?" are judged by their alphabet). The app's fixed replies are translated into Hindi, Spanish, French, German, Portuguese and Italian.

The sidebar keeps a list of conversations, one per video. **View Transcript** shows the full timestamped transcript, and **View Retrieved Context** shows exactly which chunks the last answer was based on. **Export Notes** downloads the conversation as Markdown.

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in at least `HF_TOKEN` (a Hugging Face token with "Make calls to Inference Providers" permission). `.env` is never committed or copied into the Docker image.

```bash
python app_yt.py
```

Open http://127.0.0.1:7860. The embedder and reranker download on first start (they are cached afterwards) and load in the background while the page opens.

## Settings

Every setting lives in `.env`; `.env.example` lists them all with their defaults. The main ones:

| Setting | Default | What it does |
| --- | --- | --- |
| `LLM_PROVIDER` | `huggingface` | `huggingface` or `ollama` (local). There is no automatic fallback. |
| `HF_LLM_MODEL` | `openai/gpt-oss-20b` | Hugging Face answer model. |
| `LLM_MAX_TOKENS` | `1536` | Answer length limit. Reasoning models use part of it for thinking; raise it if the log warns that an answer was cut off. |
| `HF_VERIFIER_MODEL` / `OLLAMA_VERIFIER_MODEL` | unset | A second model that checks the answers. Unset, the answer model checks itself (the log says so at startup). |
| `DATABASE_URL` | unset | PostgreSQL connection; without it, conversations are lost when the app restarts. |
| `YOUTUBE_PROXY` | unset | HTTP(S) proxy for YouTube, for servers that YouTube blocks. |
| `LOG_LEVEL` | `INFO` | `WARNING` for fewer log lines, `DEBUG` for more. |

Usage limits protect your Hugging Face credits and the server: questions up to 1,000 characters, videos up to 180 minutes, and per browser 20 video loads and 120 questions per hour. Sessions idle for 2 hours are removed, and at most 50 sessions of 10 conversations each are kept in memory. All of these are adjustable in `.env`.

### Saving conversations in PostgreSQL

Create the database once (in `psql` as the `postgres` user):

```sql
CREATE USER ytchat WITH PASSWORD 'your_password';
CREATE DATABASE ytchat OWNER ytchat;
```

Then set `DATABASE_URL=postgresql://ytchat:your_password@localhost:5432/ytchat`. The app creates its tables on startup and deletes conversations untouched for `CHAT_RETENTION_DAYS` (30). Your browser remembers its session, so after a restart or refresh your conversations reappear; the first question in a restored conversation takes a few seconds longer while its index is rebuilt.

### Local models with Ollama

Install Ollama from https://ollama.com/download, then:

```bash
ollama pull qwen2.5:1.5b
pip install -r requirements-ollama.txt
```

and set `LLM_PROVIDER=ollama` (optionally `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_NUM_CTX`). `HF_TOKEN` is not needed. Small models are slow on CPU and often answer "not enough information"; a 7B model gives much better results if your machine can run it.

### Optional local verifier

`GROUNDING_CHECKER=local_ensemble` checks answers with two local classifiers (HHEM and an NLI model) instead of the language model. Fetch their weights (about 700 MB, pinned versions, checked by hash) with `python download_models.py`. The small HHEM code files in `local_models/hhem-2.1-open/` are pinned by SHA-256 in `grounding_ensemble.py`, and the app refuses to load them if they change.

## Run with Docker

```bash
docker build -t youtube-chat .
docker run --rm -p 7860:7860 --env-file .env -v hf-cache:/app/.cache/huggingface youtube-chat
```

The `hf-cache` volume keeps downloaded models between runs.

- PostgreSQL on your computer: inside the container use `host.docker.internal` instead of `localhost` in `DATABASE_URL`.
- Ollama: build with `--build-arg INSTALL_OLLAMA=true` and run with `-e LLM_PROVIDER=ollama -e OLLAMA_BASE_URL=http://host.docker.internal:11434` (on Linux also `--add-host=host.docker.internal:host-gateway`).
- Local verifier: run `python download_models.py`, then mount the weights with `-v "$PWD/local_models:/app/local_models:ro"`.

## Troubleshooting

Answer errors are shown in the chat (and not saved to the conversation), and the server log records the stage that failed and the HTTP status.

- **402 Payment Required:** the Hugging Face account has no inference credits. Add them under Settings → Billing, or switch to Ollama.
- **401 / 403:** replace `HF_TOKEN`, or enable "Make calls to Inference Providers" on it. `python check_inference.py` tests the Hugging Face connection without printing the token.
- **"YouTube blocked this server":** common on cloud hosts. Run the app on your own computer or set `YOUTUBE_PROXY`.
- **"No transcript":** the video has no captions, so it can't be used.

Restart the app after changing `.env`.

Each log line has a time, a level and a short id for the question or video load it belongs to, plus the time spent in each stage:

```
2026-10-27 03:13:53 INFO    [cd0e3045] ytchat.app: question answered in 6.2s, result verified, stages rewriting 0.8s, retrieving 0.1s, reranking 0.4s, generating 3.9s, verifying 1.0s
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests live in `tests/` and run in seconds without downloading or calling any model: `tests/conftest.py` swaps in small fakes (set `YTCHAT_REAL_MODELS=1` to use the real ones). The PostgreSQL tests run only when `TEST_DATABASE_URL` points to a throwaway database, because they empty its tables. GitHub Actions runs the whole suite, including PostgreSQL, on every push (`.github/workflows/tests.yml`).

`scripts/` holds older manual end-to-end runners, and `evaluate_factual_support.py` benchmarks the local classifier on `benchmarks/factual_support_cases.json`.

## Project layout

| Path | Contents |
| --- | --- |
| `app_yt.py` | Settings, models, retrieval, prompts, sessions and the Gradio UI |
| `youtube_source.py` | Everything that talks to YouTube |
| `grounding.py`, `grounding_ensemble.py` | Answer checking, citations and translated fixed replies |
| `chat_store.py` | PostgreSQL storage |
| `app_logging.py`, `system_memory.py`, `inference_errors.py` | Logging, RAM checks, readable model errors |
| `app_style.css` | Page styling, with light and dark themes |
| `download_models.py` | Fetches the optional local verifier weights |
| `tests/`, `scripts/`, `benchmarks/` | Automated tests, manual runners, benchmark data |

`app_yt.py` is formatted with `black` (settings in `pyproject.toml`).

## Version control

`.gitignore` keeps out `.env`, `.venv/`, model weights, old `.bak`/`.orig` copies, `verification_logs/` and `project_snapshots/`. To start the repository once:

```bash
git init
git add .
git commit -m "Initial commit"
```

If git asks who you are, run `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"` first. Check `git status` before each commit; `.env` must never appear in it.

## Security

- Secrets stay in `.env`; error messages and logs never include tokens or request contents.
- Transcripts, titles and descriptions are treated as untrusted: they can't give the model instructions, and they're shown as plain text, so a video can't add links, images or formatting to the page. Links the model writes are removed; the only links shown are the app's own timestamp links.
- The retrieved-context viewer shows the transcript chunks behind an answer, by design; it shows no secrets or prompts beyond that context.

## Limitations

- Answer checking uses a model, so it can still be wrong, and small models often refuse questions the video does answer.
- Search indexes live in memory, one per loaded video, in a single app process.
- Transcript availability and yt-dlp metadata depend on YouTube.

## Author

Maintained by [akshaya3790](https://github.com/akshaya3790).

## License

MIT
