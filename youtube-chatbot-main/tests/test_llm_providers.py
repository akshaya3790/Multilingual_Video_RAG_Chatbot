"""Mocked provider tests; these do not verify a real Ollama server or model.

Execute the actual initialization statements, prompts, chains and chat functions
from app_yt.py without loading embedding/reranker weights or building the UI.
The unchanged existing suites cover the full app and local retrieval models.
"""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import shutil
import html
import threading
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from inference_errors import generation_error_message
import logging
from app_logging import new_request_id


SOURCE = ast.parse((Path(__file__).resolve().parent.parent / "app_yt.py").read_text(encoding="utf-8"))


def mocked_app(config, failure=None, dependency_present=True):
    """Use source statements verbatim, substituting only provider/dependency I/O."""
    received = []

    def respond(prompt):
        received.append(prompt.to_string())
        if failure:
            raise failure
        if 'You check an answer against transcript quotations' in received[-1]:
            return AIMessage(content=json.dumps({'supported': True, 'answers_question': True, 'evidence_indices': [0]}))
        text = "What is RAG?" if "Standalone question:" in received[-1] else "RAG combines retrieval and generation."
        return AIMessage(content=text)

    endpoint = Mock(name="HuggingFaceEndpoint")
    hf_chat = Mock(name="ChatHuggingFace", return_value=RunnableLambda(respond))
    ollama_chat = Mock(name="ChatOllama", return_value=RunnableLambda(respond))
    module = ModuleType("langchain_ollama")
    module.ChatOllama = ollama_chat
    scope = dict(
        os=os, re=re, json=json, tempfile=tempfile, datetime=datetime,
        HuggingFaceEndpoint=endpoint, ChatHuggingFace=hf_chat,
        ChatPromptTemplate=ChatPromptTemplate, StrOutputParser=StrOutputParser,
        generation_error_message=generation_error_message,
        detect_language=Mock(return_value="en"), detect_metadata_intent=Mock(return_value=None),
        SESSION_STORE={}, gr=Mock(), time=time, threading=threading, CHAT_STORE=None, html=html,
        MAX_QUESTION_CHARS=1000, MAX_QUESTIONS_PER_HOUR=120, shutil=shutil, EXPORT_MAX_AGE=3600,
        new_request_id=new_request_id, log=logging.getLogger('ytchat.app'),
    )
    names = {
        "HF_TOKEN", "LLM_PROVIDER", "LLM_MODEL", "rewrite_prompt", "answer_prompt",
        "rewrite_chain", "answer_chain", "SESSION_LOCK", "SESSION_EXPIRED_MESSAGE",
        "VERIFIER_MODEL", "verifier_llm", "LLM_MAX_TOKENS",
    }
    functions = {
        "answer_question", "send_message", "format_history",
        "format_source_references_markdown", "export_conversation_notes", "get_session",
        "load_saved_chats", "save_chat", "allow_request", "md_text",
    }
    statements = []
    for node in SOURCE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            statements.append(node)
        elif isinstance(node, ast.If) and any(
            isinstance(child, ast.Name) and child.id == "LLM_PROVIDER"
            for child in ast.walk(node.test)
        ):
            statements.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            statements.append(node)
    with patch.dict(os.environ, config, clear=True), patch.dict(
        sys.modules, {"langchain_ollama": module if dependency_present else None}
    ):
        exec(compile(ast.Module(body=statements, type_ignores=[]), "app_yt.py", "exec"), scope)
    return scope, endpoint, hf_chat, ollama_chat, received


class TestLLMProviders(unittest.TestCase):
    def test_huggingface_remains_default_and_keeps_generation_settings(self):
        app, endpoint, hf, ollama, _ = mocked_app({"HF_TOKEN": "unit-test-token"})
        self.assertEqual(app["LLM_PROVIDER"], "huggingface")
        endpoint.assert_called_once_with(
            repo_id="openai/gpt-oss-20b", task="text-generation", max_new_tokens=1536,
            temperature=0.1, huggingfacehub_api_token="unit-test-token",
        )
        hf.assert_called_once_with(llm=endpoint.return_value)
        ollama.assert_not_called()

    def test_huggingface_custom_model_and_missing_token(self):
        _, endpoint, _, _, _ = mocked_app({"HF_TOKEN": "unit-test-token", "HF_LLM_MODEL": "custom/model"})
        self.assertEqual(endpoint.call_args.kwargs["repo_id"], "custom/model")
        with self.assertRaisesRegex(ValueError, "HF_TOKEN is missing"):
            mocked_app({})

    def test_ollama_needs_no_hf_token_and_never_constructs_hf_inference(self):
        app, endpoint, hf, ollama, _ = mocked_app({"LLM_PROVIDER": "ollama"})
        self.assertEqual(app["LLM_PROVIDER"], "ollama")
        endpoint.assert_not_called()
        hf.assert_not_called()
        ollama.assert_called_once_with(
            model="qwen2.5:1.5b", base_url="http://127.0.0.1:11434",
            temperature=0.1, num_predict=1536, num_ctx=8192, client_kwargs={"timeout": 120.0},
        )

    def test_ollama_configuration_overrides(self):
        _, _, _, ollama, _ = mocked_app({
            "LLM_PROVIDER": " OLLAMA ", "OLLAMA_MODEL": "custom:tag",
            "OLLAMA_BASE_URL": "http://localhost:11434", "OLLAMA_NUM_CTX": "4096",
        })
        self.assertEqual(ollama.call_args.kwargs["model"], "custom:tag")
        self.assertEqual(ollama.call_args.kwargs["base_url"], "http://localhost:11434")
        self.assertEqual(ollama.call_args.kwargs["num_ctx"], 4096)

    def test_invalid_provider_context_and_missing_dependency_are_clear(self):
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            mocked_app({"LLM_PROVIDER": "unknown"})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            mocked_app({"LLM_PROVIDER": "ollama", "OLLAMA_NUM_CTX": "0"})
        with self.assertRaisesRegex(ImportError, "requirements-ollama.txt"):
            mocked_app({"LLM_PROVIDER": "ollama"}, dependency_present=False)

    def test_existing_chains_receive_context_history_metadata_and_return_strings(self):
        app, endpoint, hf, _, received = mocked_app({"LLM_PROVIDER": "ollama"})
        doc = Document(page_content="RAG combines retrieval and generation.")
        chat = dict(
            messages=[{"role": "user", "content": "Earlier question"}],
            metadata={"title": "RAG lesson"}, language="en", retriever=Mock(),
            video_id="dQw4w9WgXcQ",
            transcript_entries=[{"text": doc.page_content, "start": 12.8}],
        )
        chat["retriever"].invoke.return_value = [doc]
        app["rerank_documents"] = Mock(return_value=[doc])
        answer = app["answer_question"]("What does it combine?", chat)
        self.assertIsInstance(answer, str)
        self.assertIn("RAG combines retrieval and generation.", answer)
        self.assertIn("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=12s", answer)
        self.assertEqual(len(received), 3)
        self.assertIn("Earlier question", received[0])
        for expected in [doc.page_content, "RAG lesson", "Earlier question", "What does it combine?", "en"]:
            self.assertIn(expected, received[1])
        self.assertIn(doc.page_content, received[2])
        self.assertIn('quotations', received[2])
        chat["retriever"].invoke.assert_called_once_with("What is RAG?")
        app["rerank_documents"].assert_called_once_with("What is RAG?", [doc], top_n=4)
        endpoint.assert_not_called()
        hf.assert_not_called()

    def test_token_budget_is_configurable(self):
        _, endpoint, _, _, _ = mocked_app({"HF_TOKEN": "t", "LLM_MAX_TOKENS": "3000"})
        self.assertEqual(endpoint.call_args.kwargs["max_new_tokens"], 3000)
        self.assertNotIn("do_sample", endpoint.call_args.kwargs)
        _, _, _, ollama, _ = mocked_app({"LLM_PROVIDER": "ollama", "LLM_MAX_TOKENS": "800"})
        self.assertEqual(ollama.call_args.kwargs["num_predict"], 800)

    def test_old_saved_errors_are_left_out_of_the_prompt_history(self):
        app, *_ = mocked_app({"LLM_PROVIDER": "ollama"})
        history = app["format_history"]([
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "⚠️ **Generation Error:** credits are unavailable."},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "A real answer."},
        ])
        self.assertNotIn("Generation Error", history)
        self.assertIn("A real answer.", history)

    def test_separate_checker_model_is_built_only_when_configured(self):
        app, endpoint, hf, _, _ = mocked_app({"HF_TOKEN": "t", "HF_VERIFIER_MODEL": "Qwen/Qwen2.5-7B-Instruct"})
        self.assertEqual([call.kwargs["repo_id"] for call in endpoint.call_args_list],
                         ["openai/gpt-oss-20b", "Qwen/Qwen2.5-7B-Instruct"])
        self.assertIsNotNone(app["verifier_llm"])
        app, _, _, ollama, _ = mocked_app({"LLM_PROVIDER": "ollama", "OLLAMA_VERIFIER_MODEL": "llama3.2:3b"})
        self.assertEqual([call.kwargs["model"] for call in ollama.call_args_list], ["qwen2.5:1.5b", "llama3.2:3b"])
        # Naming the answer model again is the same as not setting it.
        app, _, _, ollama, _ = mocked_app({"LLM_PROVIDER": "ollama", "OLLAMA_VERIFIER_MODEL": "qwen2.5:1.5b"})
        self.assertEqual(ollama.call_count, 1)
        self.assertIsNone(app["verifier_llm"])

    def test_answer_is_checked_by_the_checker_model_not_the_writer(self):
        app, _, _, _, received = mocked_app({"LLM_PROVIDER": "ollama", "OLLAMA_VERIFIER_MODEL": "llama3.2:3b"})
        checked = []

        def checker(prompt):
            checked.append(prompt.to_string())
            return AIMessage(content=json.dumps({"supported": True, "answers_question": True, "evidence_indices": [0]}))

        app["verifier_llm"] = RunnableLambda(checker)
        doc = Document(page_content="RAG combines retrieval and generation.")
        chat = dict(messages=[], metadata={"title": "RAG"}, language="en", retriever=Mock(),
                    video_id="dQw4w9WgXcQ", transcript_entries=[{"text": doc.page_content, "start": 1.0}])
        chat["retriever"].invoke.return_value = [doc]
        app["rerank_documents"] = Mock(return_value=[doc])
        answer = app["answer_question"]("What does RAG combine?", chat)
        self.assertIn("RAG combines retrieval and generation.", answer)
        self.assertEqual(len(checked), 1)
        self.assertIn("quotations", checked[0])
        self.assertFalse(any("You check an answer" in prompt for prompt in received))

    def test_ollama_failures_keep_history_and_do_not_fall_back(self):
        for status, expected in [(None, "Start Ollama"), (404, "ollama pull"), (500, "server log")]:
            with self.subTest(status=status):
                exc = ConnectionError("secret") if status is None else RuntimeError("secret")
                if status:
                    exc.status_code = status
                app, endpoint, hf, _, _ = mocked_app({"LLM_PROVIDER": "ollama"}, failure=exc)
                previous = [{"role": "assistant", "content": "Earlier answer"}]
                chat = dict(messages=previous.copy(), metadata={}, language="en", retriever=Mock(),
                            _last_references=[{"seconds": 99}])
                app["SESSION_STORE"]["session"] = {"chats": {"chat": chat}}
                output, field = app["send_message"]("session", "chat", "A transcript question", previous.copy())
                self.assertIn(expected, output[-1]["content"])
                self.assertNotIn("secret", output[-1]["content"])
                self.assertEqual(output[0], previous[0])
                self.assertEqual(chat["messages"], previous)  # the error is shown, not saved
                self.assertEqual(chat["_last_references"], [])
                self.assertEqual(output[-2]["content"], "A transcript question")
                self.assertEqual(field, "A transcript question")  # kept so it can be resent
                endpoint.assert_not_called()
                hf.assert_not_called()

    def test_mocked_ollama_answer_and_references_survive_existing_export(self):
        app, _, _, _, _ = mocked_app({"LLM_PROVIDER": "ollama"})
        reference = {"timestamp_str": "00:12", "seconds": 12, "excerpt": "RAG explanation",
                     "link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=12s"}
        answer = app["answer_chain"].invoke(dict(metadata="{}", context="RAG explanation",
                    history="No previous conversation.", question="What is RAG?", answer_language="en"))
        chat = dict(metadata={"title": "Provider export test"}, video_id="dQw4w9WgXcQ",
                    messages=[{"role": "user", "content": "What is RAG?"},
                              {"role": "assistant", "content": answer, "references": [reference]}])
        app["SESSION_STORE"]["session"] = {"chats": {"chat": chat}}
        with tempfile.TemporaryDirectory() as directory, patch("tempfile.gettempdir", return_value=directory):
            path = app["export_conversation_notes"]("session", "chat")
            exported = Path(path).read_text(encoding="utf-8")
        self.assertIn("What is RAG?", exported)
        self.assertIn(answer, exported)
        self.assertIn(reference["link"], exported)
        self.assertEqual(len(chat["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
