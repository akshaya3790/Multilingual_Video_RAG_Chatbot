"""PostgreSQL chat storage. Needs a throwaway database:

    set TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_db
    python -m unittest test_chat_store

Skipped when TEST_DATABASE_URL is not set. The tests empty the chats table.
"""
import os
import unittest
import uuid
from unittest.mock import Mock, patch

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "TEST_DATABASE_URL is not set")
class TestChatStore(unittest.TestCase):
    def setUp(self):
        import psycopg
        import app_yt as app
        from chat_store import ChatStore
        self.app = app
        self.store = ChatStore(TEST_DATABASE_URL)
        with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
            conn.execute("DELETE FROM chats")
        self.saved_sessions = dict(app.SESSION_STORE)
        app.SESSION_STORE.clear()
        self.patches = [patch.object(app, "CHAT_STORE", self.store)]
        for p in self.patches:
            p.start()
        self.session = str(uuid.uuid4())

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.app.SESSION_STORE.clear()
        self.app.SESSION_STORE.update(self.saved_sessions)

    def load(self, title="Video", url="https://youtu.be/dQw4w9WgXcQ"):
        app = self.app
        with patch.object(app, "get_video_metadata", return_value={"title": title, "channel": "Channel", "url": url}), \
             patch.object(app, "get_transcript", return_value=("Saved transcript", "en", "en", [{"text": "Saved transcript", "start": 1.0}])), \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()):
            return app.load_video(self.session, url)[-1]

    def restart(self):
        """Forget everything in memory, as a server restart does."""
        self.app.SESSION_STORE.clear()

    def test_saved_chat_never_contains_the_index_or_debug_state(self):
        chat = {"id": "c1", "retriever": object(), "_last_references": [1], "messages": [], "title": "T"}
        self.store.save_chat(self.session, chat)
        restored = self.store.load_chats(self.session)
        self.assertEqual(restored, [{"id": "c1", "messages": [], "title": "T", "retriever": None}])

    def test_conversation_survives_a_restart_and_keeps_answering(self):
        app = self.app
        chat_id = self.load()
        with patch.object(app, "answer_question", return_value="First answer"):
            app.send_message(self.session, chat_id, "First question", [])
        self.restart()

        messages, status = app.select_chat(self.session, chat_id)
        self.assertEqual([m["content"] for m in messages], ["First question", "First answer"])
        self.assertIn("Video", status)
        chat = app.SESSION_STORE[self.session]["chats"][chat_id]
        self.assertIsNone(chat["retriever"])
        self.assertEqual(chat["transcript_entries"], [{"text": "Saved transcript", "start": 1.0}])

        # The first question after a restart rebuilds the search index from the saved transcript.
        rebuilt = Mock(invoke=Mock(return_value=[]))
        with patch.object(app, "build_hybrid_retriever", return_value=rebuilt) as build, \
             patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Second question"))):
            chatbot, _ = app.send_message(self.session, chat_id, "Second question", messages)
        build.assert_called_once_with("Saved transcript", "dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")
        self.assertIn("couldn't find relevant", chatbot[-1]["content"])
        self.restart()
        self.assertEqual(len(app.get_session(self.session)["chats"][chat_id]["messages"]), 4)

    def test_browser_reconnect_gets_its_sidebar_back(self):
        chat_id = self.load(title="Saved video")
        self.restart()
        session_id, choices = self.app.start_session(self.session)
        self.assertEqual(session_id, self.session)
        self.assertEqual(choices["choices"], [("Saved video", chat_id)])

    def test_invalid_saved_id_starts_a_new_session(self):
        session_id, choices = self.app.start_session("not-a-uuid")
        self.assertNotEqual(session_id, "not-a-uuid")
        self.assertEqual(choices["choices"], [])

    def test_other_sessions_cannot_read_or_overwrite_a_chat(self):
        chat_id = self.load()
        other = str(uuid.uuid4())
        self.assertEqual(self.store.load_chats(other), [])
        self.store.save_chat(other, {"id": chat_id, "messages": ["hijack"]})
        self.restart()
        self.assertEqual(self.app.get_session(self.session)["chats"][chat_id]["messages"], [])

    def test_clearing_and_trimming_are_saved(self):
        app = self.app
        first = self.load(title="First")
        with patch.object(app, "answer_question", return_value="Answer"):
            app.send_message(self.session, first, "Question", [])
        app.clear_current_chat(self.session, first)
        with patch.object(app, "MAX_CHATS_PER_SESSION", 1):
            second = self.load(title="Second")
        self.restart()
        self.assertEqual(list(app.get_session(self.session)["chats"]), [second])

    def test_database_failure_does_not_break_the_chat(self):
        app = self.app
        chat_id = self.load()
        broken = Mock(save_chat=Mock(side_effect=OSError("database down")))
        with patch.object(app, "CHAT_STORE", broken), \
             patch.object(app, "answer_question", return_value="Still answers"):
            chatbot, _ = app.send_message(self.session, chat_id, "Question", [])
        self.assertEqual(chatbot[-1]["content"], "Still answers")


if __name__ == "__main__":
    unittest.main()
