import unittest
import os
import tempfile
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
import app_yt


class TestThreeFeatures(unittest.TestCase):

    def test_authentication_failure_is_actionable_and_clears_old_citations(self):
        from types import SimpleNamespace
        session_id = app_yt.create_session()
        chat = {"messages": [], "_last_references": [{"start": 10}]}
        app_yt.SESSION_STORE[session_id]["chats"]["auth-test"] = chat
        exc = RuntimeError("private-request-data")
        exc.response = SimpleNamespace(status_code=401)
        try:
            with patch("app_yt.answer_question", side_effect=exc):
                output, _ = app_yt.send_message(
                    session_id, "auth-test", "What is the video about?", []
                )
            self.assertIn("Replace HF_TOKEN", output[-1]["content"])
            self.assertNotIn("private-request-data", output[-1]["content"])
            self.assertEqual(chat["messages"], [])  # the error is not saved to history
            self.assertEqual(chat["_last_references"], [])
        finally:
            del app_yt.SESSION_STORE[session_id]

    # ========================================================
    # FEATURE 1: TIMESTAMP-BASED ANSWERS & SOURCE REFERENCES
    # ========================================================

    # find_source_references was unused and has been removed; these cases now
    # cover source_references_for_evidence, which builds the links users see.
    def test_source_references_valid(self):
        from grounding import source_references_for_evidence
        video_id = "testVid1234"  # real YouTube ids are 11 characters
        transcript_entries = [
            {"text": "Retrieval Augmented Generation combines search with LLMs.", "start": 83.5, "duration": 4.0},
            {"text": "Vector databases store dense embeddings for semantic search.", "start": 145.0, "duration": 5.0},
            {"text": "BM25 provides lexical matching for exact keywords.", "start": 210.2, "duration": 4.5},
        ]
        evidence = ["Retrieval Augmented Generation combines search with LLMs.",
                    "BM25 provides lexical matching for exact keywords."]

        refs = source_references_for_evidence(evidence, transcript_entries, video_id, max_refs=3)
        self.assertEqual(len(refs), 2)

        self.assertEqual(refs[0]["seconds"], 83)
        self.assertEqual(refs[0]["timestamp_str"], "01:23")
        self.assertEqual(refs[0]["link"], f"https://www.youtube.com/watch?v={video_id}&t=83s")
        self.assertIn("Retrieval Augmented Generation", refs[0]["excerpt"])

        self.assertEqual(refs[1]["seconds"], 210)
        self.assertEqual(refs[1]["timestamp_str"], "03:30")
        self.assertEqual(refs[1]["link"], f"https://www.youtube.com/watch?v={video_id}&t=210s")
        self.assertIn("BM25 provides lexical matching", refs[1]["excerpt"])

    def test_source_references_edge_cases(self):
        from grounding import source_references_for_evidence as refs_for
        video_id = "testVid1234"
        valid_entries = [{"text": "Sample text segment for testing.", "start": 10.0, "duration": 2.0}]
        evidence = ["Sample text segment for testing."]

        self.assertEqual(refs_for([], valid_entries, video_id), [])
        self.assertEqual(refs_for(evidence, [], video_id), [])
        self.assertEqual(refs_for(evidence, valid_entries, ""), [])
        self.assertEqual(refs_for(evidence, valid_entries, None), [])

        bad_entries = [{"text": "Sample text segment for testing.", "start": "not-a-number", "duration": 2.0}]
        self.assertEqual(refs_for(evidence, bad_entries, video_id), [])

        # A negative start is invalid data, so it gets no link rather than a guessed 00:00.
        neg_entries = [{"text": "Sample text segment for testing.", "start": -15.0, "duration": 2.0}]
        self.assertEqual(refs_for(evidence, neg_entries, video_id), [])
        self.assertEqual(len(refs_for(evidence, valid_entries, video_id)), 1)

        unrelated = refs_for(["Completely unrelated topic about culinary recipes."], valid_entries, video_id)
        self.assertEqual(unrelated, [])

    def test_format_source_references_markdown(self):
        refs = [
            {
                "timestamp_str": "01:23",
                "seconds": 83,
                "excerpt": "Retrieval Augmented Generation combines search with LLMs.",
                "link": "https://www.youtube.com/watch?v=abc12345678&t=83s"
            }
        ]
        md = app_yt.format_source_references_markdown(refs)
        self.assertIn("<details open class='source-references-box'>", md)
        self.assertIn("📍 Source References & Relevant Video Moments", md)
        self.assertIn("[01:23](https://www.youtube.com/watch?v=abc12345678&t=83s)", md)
        self.assertIn("Retrieval Augmented Generation", md)

        # Empty refs returns empty string
        self.assertEqual(app_yt.format_source_references_markdown([]), "")

    def test_format_history_cleans_details(self):
        messages = [
            {"role": "user", "content": "What is RAG?"},
            {
                "role": "assistant",
                "content": "RAG is retrieval augmented generation.\n\n<details open class='source-references-box'><summary>Refs</summary>• link</details>"
            }
        ]
        history = app_yt.format_history(messages)
        self.assertIn("user: What is RAG?", history)
        self.assertIn("assistant: RAG is retrieval augmented generation.", history)
        self.assertNotIn("<details", history)
        self.assertNotIn("Refs", history)

    def test_get_transcript_always_returns_text_language_code_and_entries(self):
        mock_api = MagicMock()
        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.fetch.return_value.to_raw_data.return_value = [
            {"text": "Hello world segment.", "start": 0.0, "duration": 2.0}
        ]
        mock_api.list.return_value = [mock_transcript]

        with patch("youtube_source.YouTubeTranscriptApi", return_value=mock_api):
            text, _, code, entries = app_yt.get_transcript("abc12345678")
        self.assertEqual(text, "Hello world segment.")
        self.assertEqual(code, "en")
        self.assertEqual(entries[0]["text"], "Hello world segment.")

    # ========================================================
    # FEATURE 2: EXPORTABLE CONVERSATION & LEARNING NOTES
    # ========================================================

    def test_export_conversation_notes_full(self):
        session_id = app_yt.create_session()
        chat_id = "test_chat_export_1"
        app_yt.SESSION_STORE[session_id]["chats"][chat_id] = {
            "id": chat_id,
            "title": "Machine Learning Overview",
            "video_id": "test_video_123",
            "metadata": {
                "title": "Machine Learning Overview",
                "channel": "Tech Academy",
                "url": "https://www.youtube.com/watch?v=test_video_123"
            },
            "language": "en",
            "messages": [
                {"role": "user", "content": "What is supervised learning?"},
                {
                    "role": "assistant",
                    "content": "Supervised learning uses labeled datasets to train algorithms.\n\n<details><summary>Refs</summary>• [02:15](https://www.youtube.com/watch?v=test_video_123&t=135s)</details>",
                    "references": [
                        {
                            "timestamp_str": "02:15",
                            "seconds": 135,
                            "excerpt": "Supervised learning uses labeled datasets.",
                            "link": "https://www.youtube.com/watch?v=test_video_123&t=135s"
                        }
                    ]
                }
            ]
        }

        # Export notes
        filepath = app_yt.export_conversation_notes(session_id, chat_id)
        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith(".md"))

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("# YouTube Chatbot — Conversation Notes", content)
        self.assertIn("**Video:** Machine Learning Overview", content)
        self.assertIn("**Source:** https://www.youtube.com/watch?v=test_video_123", content)
        self.assertIn("### User", content)
        self.assertIn("What is supervised learning?", content)
        self.assertIn("### Assistant", content)
        self.assertIn("Supervised learning uses labeled datasets to train algorithms.", content)
        self.assertIn("#### Source References", content)
        self.assertIn("[02:15](https://www.youtube.com/watch?v=test_video_123&t=135s)", content)

        # Verify session state was not modified
        self.assertEqual(len(app_yt.SESSION_STORE[session_id]["chats"][chat_id]["messages"]), 2)

    def test_export_empty_conversation(self):
        session_id = app_yt.create_session()
        chat_id = "test_chat_empty"
        app_yt.SESSION_STORE[session_id]["chats"][chat_id] = {
            "id": chat_id,
            "title": "Empty Video Chat",
            "video_id": "test_empty_123",
            "metadata": {"title": "Empty Video Chat"},
            "messages": []
        }

        filepath = app_yt.export_conversation_notes(session_id, chat_id)
        self.assertIsNotNone(filepath)
        self.assertTrue(os.path.exists(filepath))

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Empty Video Chat", content)
        self.assertIn("No messages in this conversation", content)

    def test_export_invalid_session_guard(self):
        filepath = app_yt.export_conversation_notes(None, "chat_123")
        self.assertIsNone(filepath)

        filepath2 = app_yt.export_conversation_notes("non_existent_session", "chat_123")
        self.assertIsNone(filepath2)

    # ========================================================
    # FEATURE 3: SUGGESTED QUESTIONS & CONVERSATION STARTERS
    # ========================================================

    def test_suggestion_handlers_submit_correct_text(self):
        session_id = app_yt.create_session()
        chat_id = "test_chat_sug"

        mock_chat = {
            "id": chat_id,
            "title": "Test Sug Video",
            "video_id": "vid_sug_123",
            "metadata": {"title": "Test Sug Video", "channel": "Science Today"},
            "language": "en",
            "messages": []
        }
        app_yt.SESSION_STORE[session_id]["chats"][chat_id] = mock_chat

        with patch("app_yt.answer_question", return_value="The main concepts are RAG and vectors.") as mock_aq:
            # Suggestion 1: Main concepts
            bot_out1, msg_out1 = app_yt.send_suggestion_1(session_id, chat_id, [])
            self.assertEqual(msg_out1, "")
            self.assertEqual(len(bot_out1), 2)
            self.assertEqual(bot_out1[0]["content"], "What are the main concepts discussed in this video?")
            mock_aq.assert_called_with("What are the main concepts discussed in this video?", mock_chat)

        with patch("app_yt.answer_question", return_value="Here is a simple explanation.") as mock_aq:
            # Suggestion 2: Simple explanation
            bot_out2, msg_out2 = app_yt.send_suggestion_2(session_id, chat_id, bot_out1)
            self.assertEqual(msg_out2, "")
            self.assertEqual(len(bot_out2), 4)
            self.assertEqual(bot_out2[2]["content"], "Explain the key concepts of this video in simple terms.")
            mock_aq.assert_called_with("Explain the key concepts of this video in simple terms.", mock_chat)

        with patch("app_yt.answer_question", return_value="The speaker provides three examples.") as mock_aq:
            # Suggestion 3: Examples
            bot_out3, msg_out3 = app_yt.send_suggestion_3(session_id, chat_id, bot_out2)
            self.assertEqual(msg_out3, "")
            self.assertEqual(len(bot_out3), 6)
            self.assertEqual(bot_out3[4]["content"], "What examples does the speaker provide in this video?")
            mock_aq.assert_called_with("What examples does the speaker provide in this video?", mock_chat)

        with patch("app_yt.answer_question", return_value="The key takeaways are accuracy and speed.") as mock_aq:
            # Suggestion 4: Key takeaways
            bot_out4, msg_out4 = app_yt.send_suggestion_4(session_id, chat_id, bot_out3)
            self.assertEqual(msg_out4, "")
            self.assertEqual(len(bot_out4), 8)
            self.assertEqual(bot_out4[6]["content"], "What are the key takeaways from this video?")
            mock_aq.assert_called_with("What are the key takeaways from this video?", mock_chat)

    def test_suggestions_with_no_video_loaded(self):
        session_id = app_yt.create_session()
        # No chat loaded
        bot_out, msg_out = app_yt.send_suggestion_1(session_id, None, [])
        self.assertEqual(msg_out, "")
        self.assertEqual(len(bot_out), 1)
        self.assertIn("No video loaded", bot_out[0]["content"])


if __name__ == "__main__":
    unittest.main()
