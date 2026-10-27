FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Set to true to also install the optional Ollama provider.
ARG INSTALL_OLLAMA=false

WORKDIR /app

# Every dependency ships prebuilt wheels for Python 3.11, so no compiler or git is needed.
COPY requirements.txt requirements-ollama.txt ./

# CPU-only torch first, so sentence-transformers doesn't pull the multi-GB CUDA build.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    if [ "$INSTALL_OLLAMA" = "true" ]; then pip install --no-cache-dir -r requirements-ollama.txt; fi

# The app and every local module it imports.
COPY app_yt.py app_style.css app_logging.py youtube_source.py system_memory.py chat_store.py grounding.py grounding_ensemble.py factual_support.py inference_errors.py ./

# Model downloads are cached here; mount a volume to keep them between runs.
RUN useradd --create-home appuser && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "app_yt.py"]
