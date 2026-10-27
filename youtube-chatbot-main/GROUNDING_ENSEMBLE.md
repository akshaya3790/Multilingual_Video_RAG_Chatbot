# Experimental independent grounding checker

Status: NOT VERIFIED for live RAG; disabled by default. Do not treat this experiment as a completed unsupported-answer fix.

Sequential-loading update (22 October 2026): the verifier now releases HHEM
before loading DeBERTa, retaining only ordinary numeric scores between stages.
The batch API `check_many` scores all supplied pairs with HHEM first and then
scores them with DeBERTa. The existing `check` API remains available for one
pair; the current answer orchestration still calls this single-pair API, so
repeated model loading may add substantial latency. No latency improvement is
claimed. A shared lock prevents overlapping model lifetimes across ensemble
instances, and a retained model reference blocks the next load.

The 1500 MiB model-loading guard is unchanged and applies again before DeBERTa
loads; the 650 MiB and 256 MiB safety checks remain. Object cleanup does not
guarantee immediate return of allocator pages to the OS. The user-requested
2500 MiB preflight requirement still applies to investigation/live testing.
No real sequential-model test has run: latest measured free RAM was 892 MiB.
Only mocked lifecycle and orchestration tests have passed for this update.

New evidence and the frozen, unrun 30-case holdout are in
verification_logs/sequential_20261022_094650/. To undo only the sequential update,
restore grounding_ensemble.py and this document from
project_snapshots/pre_sequential_20261022_094649, and remove
test_sequential_grounding.py. Keep grounding.py and all earlier work intact.
The original simultaneous-loading findings below are historical evidence.

Existing default: `GROUNDING_CHECKER=llm`. No .env, provider, generator model/settings, retrieval, prompts, timestamp mapping, UI or session changes are needed for the default path.

The opt-in `local_ensemble` path checks literal evidence provenance, sentence-level support, independent strong contradictions, whole-answer relevance, and retained evidence support. Partial answers retain only original supported sentences and are checked again. It never silently switches providers/checkers. Missing verifier files, changed reviewed source, failed inference and malformed checking results produce a technical verification error, not an insufficient-context refusal. Ambiguous decisions report uncertainty.

## Models and scope

- HHEM-2.1-Open: vectara/hallucination_evaluation_model, revision de9a5e6ba9ba49531a022dba17e35ffeb2a6fad8, Apache 2.0. Downloaded files total 438,555,069 bytes. CPU float16.
- Foundation tokenizer/config only: google/flan-t5-base, revision 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2, 3,221,862 bytes. No foundation weights downloaded.
- Previously approved DeBERTa-xsmall: cross-encoder/nli-deberta-v3-xsmall, revision a150876415327c80daeff35ca6f68f5ed8cf5c24. CPU float32.
- Qwen remains the configured generator and performs only the separate whole-answer relevance check in this mode. Hugging Face remains available via existing provider selection. No paid API is used by the local support models.

English-only evaluation. Do not enable this experiment as a validated multilingual replacement. Evidence pairs over 512 model tokens are uncertain; no silent truncation. No new retrieval, neighbour expansion, fuzzy quote matching, automatic claim paraphrasing, or additional packages were introduced.

## Actual evidence and limitations

See verification_logs/support_repair_20261022_091111/REPORT.md and the underlying JSON/logs.

HHEM alone is insufficient: at threshold .8 it falsely accepted a stitched claim and falsely rejected supported answers. A conservative combination uses HHEM >= .8 for support unless DeBERTa contradiction >= .8; HHEM <= .1 with DeBERTa entailment < .5 is unsupported; remaining pairs are uncertain. These are experimental fixed cutoffs, not calibrated correctness probabilities.

Saved score evaluation found no false supported/unsupported decisions on 59 factual cases, but nine uncertain cases, seven of which were valid. One additional instruction-handling case is reported separately, not counted as factual entailment. HHEM and DeBERTa were run in separate processes for that score evaluation; this does not establish concurrent memory use or end-to-end accuracy. The older 40 fixtures were already used in development. The new 20 fixtures were fixed before this run, but their results are now known and they are no longer an untouched holdout for subsequent tuning.

The actual combined answer-path test stopped at the memory guard: 1424 MiB free, minimum 1500 MiB before verifier loading. No inference with both models loaded together succeeded in that test. Do not lower the guard merely to make testing pass. Allow additional headroom for the running app's embeddings, reranker, indexes and conversations. No server was restarted or process terminated.

## Manual enablement for a later controlled test

Only after freeing sufficient RAM and accepting that this is an unverified experimental path, in a new PowerShell terminal:

```powershell
Set-Location 'C:\Users\prama\Downloads\youtube-chatbot-main\youtube-chatbot-main'
$env:GROUNDING_CHECKER='local_ensemble'
$env:LLM_PROVIDER='ollama'
$env:OLLAMA_MODEL='qwen2.5:1.5b'
$env:GRADIO_SERVER_NAME='127.0.0.1'
$env:GRADIO_SERVER_PORT='7863'
.\.venv\Scripts\python.exe app_yt.py
```

Use an unoccupied port. These commands were not executed to start an application during this experiment. Existing ports 7860/7861 were untouched. Missing local files produce a technical error; application requests never download models automatically. Downloads already completed with user approval; their exact scripts and revision records are retained in the evidence folder.

To disable: stop only the app you started, remove the process-local setting with `Remove-Item Env:GROUNDING_CHECKER -ErrorAction SilentlyContinue`, and start normally. No persistent configuration was changed.

To revert source changes only: restore grounding.py from the uniquely named pre_support_repair_20261022_091111 snapshot, and remove the newly added grounding_ensemble.py, test_grounding_ensemble.py and this document. Preserve earlier factual_support.py and its tests. Keep verification evidence and existing backups. No app_yt.py restoration is required.
