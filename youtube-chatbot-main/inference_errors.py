"""Actionable, credential-free messages for inference failures."""


def generation_error_message(exc, provider="huggingface"):
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    prefix = '⚠️ **Generation Error:** '
    if provider == "ollama":
        # Ollama's ResponseError exposes status_code directly; httpx uses response.
        status = getattr(exc, "status_code", None) or status
        if status == 404:
            return prefix + (
                'The configured Ollama model is unavailable. Run ollama list and '
                'download the model named in OLLAMA_MODEL using ollama pull, then retry.'
            )
        if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in (
            'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout',
        ):
            return prefix + (
                'Could not connect to Ollama or the request timed out. Start Ollama '
                '(ollama serve if needed), check OLLAMA_BASE_URL, and retry. '
                'For slow responses, check available memory and the selected model.'
            )
        return prefix + (
            'Ollama could not generate an answer. Check the local Ollama server log, '
            'the configured model, and available memory, then retry.'
        )
    if status == 401:
        return prefix + (
            'Hugging Face rejected the configured API token (401 Unauthorized). '
            'Replace HF_TOKEN in the app’s .env file with a valid token, then restart the app. '
            'Create a token at https://huggingface.co/settings/tokens with '
            '“Make calls to Inference Providers” permission.'
        )
    if status == 403:
        return prefix + (
            'Hugging Face denied access (403 Forbidden). Check that your token has '
            '“Make calls to Inference Providers” permission and access to the configured model, '
            'then restart the app after updating .env.'
        )
    if status == 402:
        return prefix + 'Hugging Face inference credits are unavailable. Check your account’s inference billing and credits.'
    if status == 429:
        return prefix + 'Hugging Face is rate limiting requests. Wait a moment before trying again.'
    if status in (400, 404, 422):
        return prefix + 'The inference provider rejected the model or request configuration. Check HF_LLM_MODEL and provider availability.'
    if status is not None and status >= 500:
        return prefix + 'The inference service is temporarily unavailable. Please try again shortly.'
    if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in (
        'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'InferenceTimeoutError',
    ):
        return prefix + 'Could not reach the inference service. Check your internet connection and try again.'
    return prefix + 'An unexpected error prevented the answer. Check the server log for the error type, then try again.'
