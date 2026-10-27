import unittest
import os
import app_yt

class TestBaselineFunctions(unittest.TestCase):

    def test_extract_video_id_valid(self):
        urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]
        for url, expected_id in urls:
            with self.subTest(url=url):
                self.assertEqual(app_yt.extract_video_id(url), expected_id)

    def test_extract_video_id_invalid(self):
        invalid_urls = [
            "https://google.com",
            "not_a_url",
            "https://youtube.com/watch?v=invalid",
            "",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    app_yt.extract_video_id(url)

    def test_detect_metadata_intent(self):
        self.assertEqual(app_yt.detect_metadata_intent("Which channel uploaded this video?"), "channel")
        self.assertEqual(app_yt.detect_metadata_intent("Who is the uploader?"), "channel")
        self.assertEqual(app_yt.detect_metadata_intent("What is the video title?"), "title")
        self.assertEqual(app_yt.detect_metadata_intent("Show me the description"), "description")
        self.assertEqual(app_yt.detect_metadata_intent("How long is the video?"), "duration")
        self.assertEqual(app_yt.detect_metadata_intent("How many views does it have?"), "views")
        self.assertIsNone(app_yt.detect_metadata_intent("What is the main concept explained?"))

    def test_format_history(self):
        self.assertEqual(app_yt.format_history([]), "No previous conversation.")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        formatted = app_yt.format_history(messages)
        self.assertIn("user: Hello", formatted)
        self.assertIn("assistant: Hi there", formatted)

    def test_session_lifecycle(self):
        session_id = app_yt.create_session()
        self.assertIn(session_id, app_yt.SESSION_STORE)
        session = app_yt.SESSION_STORE[session_id]
        self.assertIsNone(session["active_chat"])
        self.assertEqual(session["chats"], {})

        # Test clear chat on non-existent chat
        cleared = app_yt.clear_current_chat(session_id, "dummy_id")
        self.assertEqual(cleared, [])

        # Test new chat
        msgs, status, chat_id = app_yt.new_chat(session_id)
        self.assertEqual(msgs, [])
        self.assertIsNone(chat_id)
        self.assertIn("New conversation", status)

if __name__ == "__main__":
    unittest.main()
