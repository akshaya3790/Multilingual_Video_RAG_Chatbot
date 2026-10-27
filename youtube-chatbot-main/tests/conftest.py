"""Shared test setup: run the whole suite without downloading or loading any model.

The embedder, reranker and Hugging Face chat model are replaced by small fakes
before the app is imported, so tests are fast and work offline. Set
YTCHAT_REAL_MODELS=1 to use the real models instead.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# The app modules are in the project folder, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_TOKEN", "offline-test-token")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

if os.getenv("YTCHAT_REAL_MODELS") != "1":
    import numpy as np
    import langchain_huggingface
    import sentence_transformers
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return np.zeros(len(pairs))

    langchain_huggingface.HuggingFaceEmbeddings = lambda **kwargs: DeterministicFakeEmbedding(size=32)
    langchain_huggingface.HuggingFaceEndpoint = MagicMock()
    langchain_huggingface.ChatHuggingFace = lambda **kwargs: RunnableLambda(lambda prompt: AIMessage(content="{}"))
    sentence_transformers.CrossEncoder = FakeCrossEncoder
