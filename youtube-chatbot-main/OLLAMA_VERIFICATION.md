# Ollama provider integration — verification report

Date: 18 October 2026 (Asia/Calcutta)

**Overall: PASS WITH LIMITATIONS for the approved code changes. Live Ollama
operation and the full live acceptance criteria are NOT VERIFIED.**

No Ollama application, Python Ollama adapter, or local model was installed during
this work. No model was downloaded and no paid inference was requested. The
existing `.env` and deployment default were preserved. The setup commands below
are instructions for the user, not completed actions.

## Snapshot and source preservation — PASS

Before editing, 38 project files were copied into:

```text
C:\Users\prama\Downloads\youtube-chatbot-main\project_snapshots\pre_ollama_20261018T165113Z
```

Each copied file's SHA-256 was verified. The snapshot includes source,
configuration, tests, documentation, prior backups and verification artifacts.
`.venv`, `__pycache__`, `.git` and `.pytest_cache` were excluded; installed Python
package versions were recorded separately in `INSTALLED_PACKAGES.txt`.
`SNAPSHOT_MANIFEST.json` contains the source-file hashes.

The two original candidates, `app_yt.py.orig` and `app_yt.py.original.bak`, have the
same SHA-256:

```text
8559511dae66d89be562d80e6c17f09fdab3a1a04f996ec83288a4f4c937fdd5
```

That hash matches the prefix recorded in the earlier verification report. There
is no Git metadata or independently verified upstream revision in this checkout.
Those older candidates predate features in the current application and were not
used to restore or replace the current source. No existing backup was overwritten.

## Files changed

| File | Reason |
|---|---|
| `app_yt.py` | Select the provider at initialization, conditionally require HF_TOKEN, initialize ChatOllama, and pass the provider to error formatting |
| `inference_errors.py` | Add local connection/timeout, missing-model and generation-failure messages while preserving Hugging Face handling |
| `requirements-ollama.txt` — new | Optional supported `langchain-ollama` dependency, separate from existing requirements |
| `test_llm_providers.py` — new | Eight tests of provider selection, configuration, prompt/context delivery, export, failure behavior and isolation with mocked clients |
| `README.md` | Add local setup, configuration, troubleshooting and verification instructions |
| `OLLAMA_VERIFICATION.md` — new | This report and actual test output |

New files under `verification_logs/ollama_integration/` are evidence artifacts,
not application changes. The project snapshot is stored outside the application
folder.

The complete unified diff, including this new report, is
[source_changes.diff](verification_logs/ollama_integration/source_changes.diff).

## Scope verification — PASS

[source_integrity.json](verification_logs/ollama_integration/source_integrity.json)
records the comparison against the snapshot:

- Only the three existing files listed above changed. Every existing test,
  `.env`, `requirements.txt`, Dockerfile and original backup is byte-for-byte
  unchanged.
- All 24 existing application functions remain. Only `send_message` changed, and
  its only change is passing `provider=LLM_PROVIDER` to the error formatter.
- The complete source from `CSS =` through the end of the file is byte-for-byte
  identical, preserving UI definitions, component placement, styling, labels,
  callbacks and startup code. Browser interaction was not tested in this run.
- Prompt templates, chains, embedding/reranker initialization and chunking are
  structurally identical. All retrieval, transcript, citation, export and session
  function bodies are unchanged.
- The original Hugging Face endpoint and chat-wrapper initialization statements
  are structurally identical inside the new provider branch.
- All root Python files compiled successfully in memory; no application edits
  were made to address unrelated behavior.

The Dockerfile's existing `app.py` entrypoint was left untouched.

## Provider behavior

`LLM_PROVIDER` defaults to `huggingface`. Explicit `ollama` selection lazily
imports `ChatOllama` and does not construct Hugging Face inference clients. An
unknown provider raises a configuration error. There is no automatic fallback.

Both providers feed the existing `rewrite_prompt | llm | StrOutputParser()` and
`answer_prompt | llm | StrOutputParser()` chains. The rewrite receives the latest
question and recent history. Retrieval and reranking are unchanged. Answer
generation receives the original metadata, retrieved context, history, question
and language fields and returns the existing string response format.

Ollama settings: temperature 0.1, output limit 512 tokens, default context 8192
tokens, request timeout 120 seconds. HF_TOKEN is required only in Hugging Face
mode. Existing local embeddings/reranking may still fetch their weights from
Hugging Face if they are not cached; this is separate from paid inference.

## Hardware and proposed model — PASS WITH LIMITATIONS

Inspected CPU: Intel Core i5-12450H, 12 logical processors. Usable RAM: 7.68 GiB.
Available RAM at inspection was approximately 0.3–0.6 GiB. GPU: NVIDIA GTX 1650,
4096 MiB VRAM, approximately 3937 MiB free; driver 581.83.

Proposed model: **qwen2.5:1.5b**, default Q4_K_M quantization, approximately
**986 MB download**. Its small size, instruction following and multilingual
support suit a cautious first trial with transcript passages. Model information:
https://ollama.com/library/qwen2.5:1.5b.

Estimated model-runtime memory budget: roughly 2–3 GiB, depending on context and
offloading, in addition to Windows and the existing retrieval models. This is an
estimate, not a measured requirement or fit guarantee. Close other applications
before testing. Actual GPU use, latency and peak memory are NOT VERIFIED. A small
model may perform worse on complex reasoning or multilingual answers. Long
histories may exceed the configured context window.

Ollama was not found on PATH or in its usual installation location. Port 11434
refused a local connection during inspection. The optional Python integration
was not installed. No real model response has been observed.

## Setup instructions — NOT EXECUTED

Install the official Windows application yourself:
https://ollama.com/download/windows.
Official instructions: https://docs.ollama.com/windows.
After installation, open a new PowerShell window:

```powershell
ollama --version
ollama list
ollama pull qwen2.5:1.5b
ollama run qwen2.5:1.5b "Reply with OK."
```

The Windows app normally serves Ollama in the background. If needed, run
`ollama serve` in another terminal; do not start a second server if one is already
listening on port 11434.

Install the optional Python adapter into the existing virtual environment and
start the existing application:

```powershell
Set-Location 'C:\Users\prama\Downloads\youtube-chatbot-main\youtube-chatbot-main'
.\.venv\Scripts\python.exe -m pip install -r requirements-ollama.txt
$env:LLM_PROVIDER = 'ollama'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:OLLAMA_MODEL = 'qwen2.5:1.5b'
$env:OLLAMA_NUM_CTX = '8192'
.\.venv\Scripts\python.exe app_yt.py
```

The LangChain supported adapter is documented at
https://docs.langchain.com/oss/python/integrations/chat/ollama.
Its real installed-package behavior is NOT VERIFIED here. Re-run the suites after
installing it, since installing a dependency can change Python package versions.

These environment variables affect only this terminal session. To restore the
existing provider, stop the app, set `$env:LLM_PROVIDER = 'huggingface'`, and start
it again. The current `.env` is unchanged. The existing `check_inference.py`
diagnostic checks Hugging Face only.

## Tests and exit codes

No existing test source or assertion was changed, removed or skipped.

| Run | Status | Exit code | Evidence |
|---|---|---:|---|
| Initial existing suites inside sandbox | NOT VERIFIED: 180-second startup timeout before test results | 124 for supervised run | `before_existing_tests.log` and `.json` |
| Initial focused tests inside sandbox | FAIL: export test blocked by sandbox temporary-file permissions | 1 | `provider_tests_attempt1.log` and `.json`; `provider_tests.log` and `.json` contain the same first attempt |
| Focused tests outside sandbox | PASS: 11 tests, including export | 0 | Terminal output showed `Ran 11 tests in 0.074s` and `OK`; these cases also pass in the final saved run |
| Unchanged existing suites outside sandbox | PASS: 26 tests | 0 | `existing_tests.log` and `.json` |
| Final combined run outside sandbox | PASS: 34 tests | 0 | `final_tests.log` and `.json`; full output below |
| Source/syntax integrity check | PASS | 0 | `source_integrity.json` and `source_changes.diff` |

The first sandbox supervisor saved the timeout result, then its console-print
step failed with a Unicode encoding error (wrapper exit 1). That console error
was separate from the recorded test timeout. The successful runs used UTF-8
output. No product change was made to address sandbox permissions or encoding.

The existing-suite diagnostic run used an unchanged-test `unittest.main` wrapper
with a 45-second traceback timer. Its exact command is in `existing_tests.json`.
The final run used the standard command below, without mocking the app's
embedding/reranking models. Process-only environment settings prevented model
downloads and disabled Gradio analytics:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:LLM_PROVIDER = 'huggingface'
.\.venv\Scripts\python.exe -B -m unittest test_llm_providers test_baseline test_regression_suite test_three_features test_inference_errors -v
```

The evidence JSON records the exact absolute executable and argument list. The
final log is [final_tests.log](verification_logs/ollama_integration/final_tests.log).
The unittest duration excludes module-import/model-loading time and is not an
inference benchmark.

## Live versus mocked coverage

| Requested check | Status | What was actually verified |
|---|---|---|
| Real local-model inference | NOT VERIFIED | No installed Ollama server/model or Python adapter; no real inference request made |
| Real YouTube transcript question through Ollama | NOT VERIFIED | No new live YouTube video loaded in this implementation run |
| Grounded timestamp citations | PASS WITH LIMITATIONS | Existing helper tests and a mocked-provider chain test preserve controlled transcript references and video/time URL construction; real-video semantic relevance is NOT VERIFIED |
| Unsupported-question behavior | NOT VERIFIED for live inference | Citation edge-case tests pass, but do not establish a real model's refusal behavior |
| Populated conversation export | PASS WITH LIMITATIONS | Real Markdown files written and checked using fixtures/mocked model output, preserving messages, answers and references |
| All four suggestions | PASS WITH LIMITATIONS | Existing handler tests verify all four questions with a simulated loaded chat and mocked answer function; browser clicks and live answers are NOT VERIFIED |
| Manual questions and session workflows | PASS WITH LIMITATIONS | Manual mocked-chain questions, session creation, New Chat, switching, Clear Current Chat and input guards pass; loading a second real video is NOT VERIFIED |
| Provider isolation | PASS WITH LIMITATIONS | Ollama branch constructs no HF inference clients in mocked initialization, chain and failure tests; real-adapter traffic isolation is NOT VERIFIED |
| Unavailable server, missing model, inference failure | PASS WITH LIMITATIONS | Simulated connection errors, 404 and 500 preserve prior messages and clear stale references without fallback; real server failures are NOT VERIFIED |
| Original Hugging Face path | PASS WITH LIMITATIONS | Full app initializes on the default HF path; mocked configuration/error tests and source comparison preserve it; no live paid inference requested |
| Existing FAISS/BM25/reranker | PASS | Existing regression test uses the real cached embedding and reranker models with a synthetic transcript |
| UI source preservation | PASS | Complete CSS/UI/callback/startup suffix is byte-identical; visual/browser behavior is NOT VERIFIED |

Citation attachment remains based on matching retrieved passages; semantic
support for each answer is not independently checked by that existing logic.
The prior acceptance script uses controlled LLM outputs and retrieval for its
citation cases. It was not treated as proof of real-model grounding, and was not
run here. Citation logic was not modified.

The successful full suite emitted the expected invalid-session Gradio warnings
and an unclosed asyncio-event-loop ResourceWarning. They did not fail tests and
were left unchanged. No local model ran, so no local-model memory issue was
observed or ruled out. The low available RAM remains a practical limitation.

The code is ready for user-managed installation and live testing. Full chatbot
acceptance must remain NOT VERIFIED until a real local-model request and the
real-video workflows succeed. Stop and report memory issues or feature failures
when performing that later verification; no unrelated fix is authorized.

## Actual final terminal output

The following is the captured final test output, followed by its actual exit code.

```text

Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 2575.19it/s]

Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]
Loading weights:   5%|▍         | 10/201 [00:00<00:02, 71.69it/s]
Loading weights:  13%|█▎        | 26/201 [00:00<00:01, 109.51it/s]
Loading weights:  19%|█▉        | 39/201 [00:00<00:01, 117.42it/s]
Loading weights:  29%|██▉       | 59/201 [00:00<00:00, 144.71it/s]
Loading weights:  37%|███▋      | 74/201 [00:00<00:00, 139.16it/s]
Loading weights: 100%|██████████| 201/201 [00:00<00:00, 343.78it/s]
test_existing_chains_receive_context_history_metadata_and_return_strings (test_llm_providers.TestLLMProviders.test_existing_chains_receive_context_history_metadata_and_return_strings) ... ok
test_huggingface_custom_model_and_missing_token (test_llm_providers.TestLLMProviders.test_huggingface_custom_model_and_missing_token) ... ok
test_huggingface_remains_default_and_keeps_generation_settings (test_llm_providers.TestLLMProviders.test_huggingface_remains_default_and_keeps_generation_settings) ... ok
test_invalid_provider_context_and_missing_dependency_are_clear (test_llm_providers.TestLLMProviders.test_invalid_provider_context_and_missing_dependency_are_clear) ... ok
test_mocked_ollama_answer_and_references_survive_existing_export (test_llm_providers.TestLLMProviders.test_mocked_ollama_answer_and_references_survive_existing_export) ... ok
test_ollama_configuration_overrides (test_llm_providers.TestLLMProviders.test_ollama_configuration_overrides) ... ok
test_ollama_failures_keep_history_and_do_not_fall_back (test_llm_providers.TestLLMProviders.test_ollama_failures_keep_history_and_do_not_fall_back) ... CHAT ERROR: ConnectionError HTTP status: None
CHAT ERROR: RuntimeError HTTP status: None
CHAT ERROR: RuntimeError HTTP status: None
ok
test_ollama_needs_no_hf_token_and_never_constructs_hf_inference (test_llm_providers.TestLLMProviders.test_ollama_needs_no_hf_token_and_never_constructs_hf_inference) ... ok
test_detect_metadata_intent (test_baseline.TestBaselineFunctions.test_detect_metadata_intent) ... ok
test_extract_video_id_invalid (test_baseline.TestBaselineFunctions.test_extract_video_id_invalid) ... ok
test_extract_video_id_valid (test_baseline.TestBaselineFunctions.test_extract_video_id_valid) ... ok
test_format_history (test_baseline.TestBaselineFunctions.test_format_history) ... ok
test_session_lifecycle (test_baseline.TestBaselineFunctions.test_session_lifecycle) ... ok
test_01_url_parsing_all_supported_formats (test_regression_suite.TestRegressionSuite.test_01_url_parsing_all_supported_formats) ... ok
test_02_url_parsing_invalid (test_regression_suite.TestRegressionSuite.test_02_url_parsing_invalid) ... ok
test_03_metadata_intent_detection (test_regression_suite.TestRegressionSuite.test_03_metadata_intent_detection) ... ok
test_04_metadata_answer_generation (test_regression_suite.TestRegressionSuite.test_04_metadata_answer_generation) ... ok
test_05_retriever_and_reranker_pipeline (test_regression_suite.TestRegressionSuite.test_05_retriever_and_reranker_pipeline) ... ok
test_06_session_lifecycle_and_state (test_regression_suite.TestRegressionSuite.test_06_session_lifecycle_and_state) ... ok
test_07_input_validation_and_guards (test_regression_suite.TestRegressionSuite.test_07_input_validation_and_guards) ... ok
test_authentication_failure_is_actionable_and_clears_old_citations (test_three_features.TestThreeFeatures.test_authentication_failure_is_actionable_and_clears_old_citations) ... CHAT ERROR: RuntimeError HTTP status: 401
ok
test_export_conversation_notes_full (test_three_features.TestThreeFeatures.test_export_conversation_notes_full) ... ok
test_export_empty_conversation (test_three_features.TestThreeFeatures.test_export_empty_conversation) ... ok
test_export_invalid_session_guard (test_three_features.TestThreeFeatures.test_export_invalid_session_guard) ... C:\Users\prama\Downloads\youtube-chatbot-main\youtube-chatbot-main\.venv\Lib\site-packages\gradio\helpers.py:1274: UserWarning: ⚠️ Please load a video and have an active conversation before exporting notes.
  warnings.warn(message)
C:\Users\prama\Downloads\youtube-chatbot-main\youtube-chatbot-main\.venv\Lib\site-packages\gradio\helpers.py:1274: UserWarning: ⚠️ Session has expired or is invalid.
  warnings.warn(message)
ok
test_find_source_references_edge_cases (test_three_features.TestThreeFeatures.test_find_source_references_edge_cases) ... ok
test_find_source_references_valid (test_three_features.TestThreeFeatures.test_find_source_references_valid) ... ok
test_format_history_cleans_details (test_three_features.TestThreeFeatures.test_format_history_cleans_details) ... ok
test_format_source_references_markdown (test_three_features.TestThreeFeatures.test_format_source_references_markdown) ... ok
test_get_transcript_backward_compatibility (test_three_features.TestThreeFeatures.test_get_transcript_backward_compatibility) ... C:\Users\prama\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\asyncio\base_events.py:693: ResourceWarning: unclosed event loop <ProactorEventLoop running=False closed=False debug=False>
  _warn(f"unclosed event loop {self!r}", ResourceWarning, source=self)
ResourceWarning: Enable tracemalloc to get the object allocation traceback
ok
test_suggestion_handlers_submit_correct_text (test_three_features.TestThreeFeatures.test_suggestion_handlers_submit_correct_text) ... ok
test_suggestions_with_no_video_loaded (test_three_features.TestThreeFeatures.test_suggestions_with_no_video_loaded) ... ok
test_network_failure (test_inference_errors.TestInferenceErrors.test_network_failure) ... ok
test_provider_failures_have_specific_recovery_steps (test_inference_errors.TestInferenceErrors.test_provider_failures_have_specific_recovery_steps) ... ok
test_unexpected_failure_does_not_expose_exception (test_inference_errors.TestInferenceErrors.test_unexpected_failure_does_not_expose_exception) ... ok

----------------------------------------------------------------------
Ran 34 tests in 0.909s

OK

EXIT CODE: 0
```
