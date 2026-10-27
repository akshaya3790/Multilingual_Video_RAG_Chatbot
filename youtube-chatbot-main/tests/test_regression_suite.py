import unittest
import app_yt
from langchain_core.documents import Document

class TestRegressionSuite(unittest.TestCase):

    def test_01_url_parsing_all_supported_formats(self):
        test_cases = [
            ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "jNQXAC9IVRw"),
            ("https://youtube.com/watch?v=jNQXAC9IVRw&feature=shared", "jNQXAC9IVRw"),
            ("http://www.youtube.com/embed/jNQXAC9IVRw", "jNQXAC9IVRw"),
            ("https://youtu.be/jNQXAC9IVRw", "jNQXAC9IVRw"),
            ("https://www.youtube.com/shorts/jNQXAC9IVRw", "jNQXAC9IVRw"),
            ("https://youtube.com/shorts/jNQXAC9IVRw?feature=share", "jNQXAC9IVRw"),
        ]
        for url, expected_id in test_cases:
            with self.subTest(url=url):
                vid = app_yt.extract_video_id(url)
                self.assertEqual(vid, expected_id)

    def test_02_url_parsing_invalid(self):
        invalid_cases = [
            "",
            "   ",
            "https://vimeo.com/12345678",
            "https://youtube.com/watch?v=short",
            "random text",
        ]
        for url in invalid_cases:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    app_yt.extract_video_id(url)

    def test_03_metadata_intent_detection(self):
        self.assertEqual(app_yt.detect_metadata_intent("Which channel uploaded this?"), "channel")
        self.assertEqual(app_yt.detect_metadata_intent("Who is the uploader?"), "channel")
        self.assertEqual(app_yt.detect_metadata_intent("What is the video title?"), "title")
        self.assertEqual(app_yt.detect_metadata_intent("What is the description?"), "description")
        self.assertEqual(app_yt.detect_metadata_intent("How long is the video?"), "duration")
        self.assertEqual(app_yt.detect_metadata_intent("How many views?"), "views")
        self.assertIsNone(app_yt.detect_metadata_intent("What is discussed in chapter 3?"))

    def test_03b_content_questions_bypass_metadata_router(self):
        # Content questions that merely contain a metadata keyword must reach the transcript.
        for question in [
            "What does she say about interviews?",
            "What are his views on AI?",
            "Why are subtitles important?",
            "Give a description of the algorithm",
            "What is the duration of the training?",
            "What's the title of the book he mentions?",
            "Which channel does he recommend for learning this?",
            "Who created the theory of relativity?",
            "How many views does he say his first video got?",
        ]:
            with self.subTest(question=question):
                self.assertIsNone(app_yt.detect_metadata_intent(question))

    def test_04_metadata_answer_generation(self):
        mock_meta = {
            "title": "Introduction to RAG",
            "channel": "AI Academy",
            "description": "A tutorial on Retrieval-Augmented Generation.",
            "duration": 630,
            "view_count": 15420
        }
        ans_channel = app_yt.answer_metadata_question("channel", mock_meta)
        self.assertIn("AI Academy", ans_channel)

        ans_title = app_yt.answer_metadata_question("title", mock_meta)
        self.assertIn("Introduction to RAG", ans_title)

        ans_duration = app_yt.answer_metadata_question("duration", mock_meta)
        self.assertIn("10 minutes and 30 seconds", ans_duration)

        ans_views = app_yt.answer_metadata_question("views", mock_meta)
        self.assertIn("15,420 views", ans_views)

    def test_05_retriever_and_reranker_pipeline(self):
        sample_transcript = (
            "Retrieval-Augmented Generation combines external document retrieval with large language models. "
            "Dense vector embeddings represent the semantic meaning of texts in high-dimensional vector spaces. "
            "BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query. "
            "Cross-encoders score pairs of queries and candidate documents directly to produce high accuracy reranking."
        )
        video_id = "test_vid_123"
        url = "https://www.youtube.com/watch?v=test_vid_123"

        # Build hybrid retriever
        retriever = app_yt.build_hybrid_retriever(sample_transcript, video_id, url)
        self.assertIsNotNone(retriever)

        # Query hybrid retriever
        candidates = retriever.invoke("What does RAG combine?")
        self.assertTrue(len(candidates) > 0)

        # Rerank documents
        reranked = app_yt.rerank_documents("What does RAG combine?", candidates, top_n=2)
        self.assertTrue(len(reranked) > 0)
        self.assertTrue(len(reranked) <= 2)

    def test_06_session_lifecycle_and_state(self):
        session_id = app_yt.create_session()
        self.assertIn(session_id, app_yt.SESSION_STORE)

        # Verify chat choices formatting
        choices_update = app_yt.get_chat_choices(session_id)
        self.assertEqual(choices_update.get("choices"), [])

        # Add simulated chat into session
        chat_id = "chat_abc"
        app_yt.SESSION_STORE[session_id]["chats"][chat_id] = {
            "id": chat_id,
            "title": "Sample Video Title",
            "created_at": "2026-10-18T12:00:00",
            "video_id": "test_123",
            "metadata": {
                "title": "Sample Video Title",
                "channel": "Tech Hub",
                "duration": 300
            },
            "language": "en",
            "retriever": None,
            "messages": [
                {"role": "user", "content": "What is this video about?"},
                {"role": "assistant", "content": "It is about RAG."}
            ]
        }
        app_yt.SESSION_STORE[session_id]["active_chat"] = chat_id

        choices_update2 = app_yt.get_chat_choices(session_id)
        self.assertEqual(choices_update2.get("choices"), [("Sample Video Title", chat_id)])

        # Select chat
        restored_msgs, status = app_yt.select_chat(session_id, chat_id)
        self.assertEqual(len(restored_msgs), 2)
        self.assertIn("Sample Video Title", status)
        self.assertIn("Tech Hub", status)

        # Clear current chat
        cleared = app_yt.clear_current_chat(session_id, chat_id)
        self.assertEqual(cleared, [])
        self.assertEqual(app_yt.SESSION_STORE[session_id]["chats"][chat_id]["messages"], [])

        # New chat
        new_msgs, new_status, new_active = app_yt.new_chat(session_id)
        self.assertEqual(new_msgs, [])
        self.assertIsNone(new_active)
        self.assertIn("New conversation", new_status)

    def test_07_input_validation_and_guards(self):
        session_id = app_yt.create_session()

        # Empty URL validation
        status, choices, bot, msg, cid = app_yt.load_video(session_id, "")
        self.assertIn("Missing URL", status)
        self.assertIsNone(cid)

        # Invalid URL validation
        status2, choices2, bot2, msg2, cid2 = app_yt.load_video(session_id, "https://not-youtube.com")
        self.assertIn("Invalid Link", status2)
        self.assertIsNone(cid2)

        # Message without active chat
        bot_out, msg_out = app_yt.send_message(session_id, None, "Hello?", [])
        self.assertEqual(msg_out, "")
        self.assertEqual(len(bot_out), 1)
        self.assertIn("No video loaded", bot_out[0]["content"])

        # Empty message
        bot_out2, msg_out2 = app_yt.send_message(session_id, "chat_abc", "   ", [])
        self.assertEqual(msg_out2, "")
        self.assertEqual(bot_out2, [])

if __name__ == "__main__":
    unittest.main()
