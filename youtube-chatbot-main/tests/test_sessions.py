"""Session limits and recovery from sessions the server no longer knows."""
import time
import unittest
from unittest.mock import Mock, patch

import app_yt as app


def fake_video(title="Video"):
    """Patch the network/index steps of load_video."""
    return (
        patch.object(app, "get_video_metadata", return_value={"title": title, "channel": "Channel"}),
        patch.object(app, "get_transcript", return_value=("Transcript", "en", "en", [])),
        patch.object(app, "build_hybrid_retriever", return_value=Mock()),
    )


class TestSessions(unittest.TestCase):
    def setUp(self):
        self.saved = dict(app.SESSION_STORE)
        app.SESSION_STORE.clear()

    def tearDown(self):
        app.SESSION_STORE.clear()
        app.SESSION_STORE.update(self.saved)

    def load(self, session_id, url="https://youtu.be/dQw4w9WgXcQ", title="Video"):
        metadata, transcript, index = fake_video(title)
        with metadata, transcript, index:
            return app.load_video(session_id, url)

    def test_stale_session_id_can_still_load_a_video(self):
        # e.g. the browser kept its id across a server restart
        result = self.load("id-from-before-restart")
        self.assertIn("ready for questions", result[0])
        self.assertIn(result[-1], app.SESSION_STORE["id-from-before-restart"]["chats"])

    def test_load_before_session_exists_explains_instead_of_failing(self):
        result = self.load(None)
        self.assertIn("Still starting up", result[0])
        self.assertIsNone(result[-1])
        self.assertEqual(app.SESSION_STORE, {})

    def test_stale_session_message_and_selection_say_session_expired(self):
        chatbot, field = app.send_message("unknown", "chat", "Hello?", [])
        self.assertIn("Session expired", chatbot[-1]["content"])
        self.assertEqual(field, "")
        messages, status = app.select_chat("unknown", "chat")
        self.assertEqual(messages, [])
        self.assertIn("Session expired", status)

    def test_idle_sessions_are_evicted(self):
        idle = app.create_session()
        app.SESSION_STORE[idle]["last_seen"] = time.monotonic() - app.SESSION_IDLE_TIMEOUT - 1
        active = app.create_session()
        self.assertNotIn(idle, app.SESSION_STORE)
        self.assertIn(active, app.SESSION_STORE)

    def test_session_being_loaded_is_not_evicted(self):
        busy = app.create_session()
        app.SESSION_STORE[busy]["last_seen"] = time.monotonic() - app.SESSION_IDLE_TIMEOUT - 1
        app.SESSION_STORE[busy]["_viewer_loading"] = True
        app.create_session()
        self.assertIn(busy, app.SESSION_STORE)

    def test_session_count_is_capped_least_recently_used_first(self):
        with patch.object(app, "MAX_SESSIONS", 3):
            ids = [app.create_session() for _ in range(3)]
            app.get_session(ids[0])  # ids[0] is now the most recently used
            newest = app.create_session()
        self.assertEqual(set(app.SESSION_STORE), {ids[0], ids[2], newest})

    def test_chats_per_session_are_capped_keeping_the_active_one(self):
        session_id = app.create_session()
        with patch.object(app, "MAX_CHATS_PER_SESSION", 2):
            first = self.load(session_id, title="First")[-1]
            second = self.load(session_id, "https://youtu.be/jNQXAC9IVRw", title="Second")[-1]
            third = self.load(session_id, title="Third")[-1]
        chats = app.SESSION_STORE[session_id]["chats"]
        self.assertEqual(list(chats), [second, third])
        self.assertNotIn(first, chats)
        self.assertEqual(app.SESSION_STORE[session_id]["active_chat"], third)

    def test_second_question_while_answering_is_refused_without_losing_it(self):
        session_id = app.create_session()
        chat_id = self.load(session_id)[-1]
        send = app.with_context_viewer(app.send_message, manual=True)
        with patch.object(app, "answer_question", return_value="Answer"), \
             patch.object(app.gr, "Warning") as warning:
            first = send(session_id, chat_id, "First question", [])
            next(first)  # the first answer is now in progress
            second = list(send(session_id, chat_id, "Second question", []))
            self.assertEqual(second, [(app.gr.skip(),) * 3])  # typed text stays in the box
            warning.assert_called_once()
            list(first)  # first answer finishes
            third = list(send(session_id, chat_id, "Third question", []))
        self.assertEqual(third[-1][0][-1]["content"], "Answer")
        messages = app.SESSION_STORE[session_id]["chats"][chat_id]["messages"]
        self.assertEqual([m["content"] for m in messages if m["role"] == "user"],
                         ["First question", "Third question"])

    def test_failed_load_deselects_sidebar_and_selecting_a_chat_reenables_typing(self):
        session_id = app.create_session()
        chat_id = self.load(session_id)[-1]
        failed = app.load_video(session_id, "https://not-youtube.com/watch")
        self.assertIn("Invalid Link", failed[0])
        self.assertIsNone(failed[1]["value"])  # the old chat can be clicked again
        self.assertFalse(failed[3]["interactive"])
        self.assertTrue(app.message_box_for_chat(session_id, chat_id)["interactive"])
        self.assertFalse(app.message_box_for_chat(session_id, None)["interactive"])


class TestUntrustedTextAndLimits(unittest.TestCase):
    def setUp(self):
        self.saved = dict(app.SESSION_STORE)
        app.SESSION_STORE.clear()
        self.session = app.create_session()

    def tearDown(self):
        app.SESSION_STORE.clear()
        app.SESSION_STORE.update(self.saved)

    def test_youtube_text_cannot_inject_links_images_or_html(self):
        title = "Win [a prize](https://evil.example) ![x](https://track.example/p.png) <b>now</b> https://evil.example **bold**"
        shown = app.md_text(title)
        for unsafe in ["](", "![", "<b>", "**", "https://"]:
            self.assertNotIn(unsafe, shown)
        self.assertIn("&lt;b&gt;", shown)
        answer = app.answer_metadata_question("title", {"title": title})
        self.assertIn(shown, answer)
        status = self.load_status(title)
        self.assertIn(shown, status)
        self.assertNotIn("[a prize](", status)

    def load_status(self, title, duration=60, transcript="Transcript"):
        with patch.object(app, "get_video_metadata", return_value={"title": title, "channel": "C", "duration": duration}), \
             patch.object(app, "get_transcript", return_value=(transcript, "en", "en", [])), \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()) as index:
            result = app.load_video(self.session, "https://youtu.be/dQw4w9WgXcQ")
        self.indexed = index.called
        return result[0]

    def test_overlong_video_and_transcript_are_refused_before_indexing(self):
        status = self.load_status("Long", duration=app.MAX_VIDEO_MINUTES * 60 + 60)
        self.assertIn("Video too long", status)
        self.assertFalse(self.indexed)
        status = self.load_status("Huge", transcript="x" * (app.MAX_TRANSCRIPT_CHARS + 1))
        self.assertIn("Transcript too long", status)
        self.assertFalse(self.indexed)

    def test_video_loads_are_rate_limited_per_session(self):
        with patch.object(app, "MAX_VIDEOS_PER_HOUR", 2):
            self.assertIn("ready", self.load_status("One"))
            self.assertIn("ready", self.load_status("Two"))
            self.assertIn("Too many videos", self.load_status("Three"))
            other = app.create_session()
            self.session = other
            self.assertIn("ready", self.load_status("Other browser"))

    def test_overlong_question_is_refused_without_calling_the_model(self):
        self.load_status("Video")
        chat_id = app.SESSION_STORE[self.session]["active_chat"]
        question = "x" * (app.MAX_QUESTION_CHARS + 1)
        with patch.object(app, "answer_question") as answer:
            chatbot, field = app.send_message(self.session, chat_id, question, [])
        answer.assert_not_called()
        self.assertIn("Question too long", chatbot[-1]["content"])
        self.assertEqual(field, question)  # kept so it can be shortened
        self.assertEqual(app.SESSION_STORE[self.session]["chats"][chat_id]["messages"], [])

    def test_questions_are_rate_limited_per_session(self):
        self.load_status("Video")
        chat_id = app.SESSION_STORE[self.session]["active_chat"]
        with patch.object(app, "MAX_QUESTIONS_PER_HOUR", 1), \
             patch.object(app, "answer_question", return_value="Answer"):
            first, _ = app.send_message(self.session, chat_id, "One", [])
            second, _ = app.send_message(self.session, chat_id, "Two", first)
        self.assertEqual(first[-1]["content"], "Answer")
        self.assertIn("Too many questions", second[-1]["content"])


class TestYouTubeErrors(unittest.TestCase):
    def setUp(self):
        self.saved = dict(app.SESSION_STORE)
        app.SESSION_STORE.clear()
        self.session = app.create_session()

    def tearDown(self):
        app.SESSION_STORE.clear()
        app.SESSION_STORE.update(self.saved)

    def status_for(self, metadata_error=None, transcript_error=None):
        metadata = {"title": "T", "channel": "C"}
        with patch.object(app, "get_video_metadata", side_effect=metadata_error, return_value=metadata), \
             patch.object(app, "get_transcript", side_effect=transcript_error, return_value=("x", "en", "en", [])), \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()):
            return app.load_video(self.session, "https://youtu.be/dQw4w9WgXcQ")[0]

    def test_blocked_server_gets_a_clear_message(self):
        from youtube_transcript_api import IpBlocked
        from yt_dlp.utils import DownloadError
        self.assertIn("YouTube blocked this server", self.status_for(transcript_error=IpBlocked("dQw4w9WgXcQ")))
        bot_check = DownloadError("ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you’re not a bot.")
        self.assertIn("YOUTUBE_PROXY", self.status_for(metadata_error=bot_check))

    def test_missing_transcript_and_private_video_are_told_apart(self):
        from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable
        from yt_dlp.utils import DownloadError
        self.assertIn("No transcript", self.status_for(transcript_error=TranscriptsDisabled("dQw4w9WgXcQ")))
        self.assertIn("No transcript", self.status_for(transcript_error=ValueError("No transcript is available for this video.")))
        self.assertIn("Video unavailable", self.status_for(transcript_error=VideoUnavailable("dQw4w9WgXcQ")))
        self.assertIn("Video unavailable", self.status_for(metadata_error=DownloadError("ERROR: Private video. Sign in if you've been granted access")))
        missing = DownloadError("ERROR: [youtube] aaaaaaaaaaa: This video is unavailable")  # real yt-dlp text
        self.assertIn("Video unavailable", self.status_for(metadata_error=missing))
        self.assertIn("Could Not Load Video", self.status_for(metadata_error=RuntimeError("something else")))

    def test_invalid_link_says_so(self):
        status = app.load_video(self.session, "https://example.com/watch?v=dQw4w9WgXcQ")[0]
        self.assertIn("Invalid Link", status)  # caught before YouTube is contacted

    def test_proxy_is_passed_to_both_youtube_clients(self):
        proxy = "http://user:pass@proxy.example:8080"
        import os
        import youtube_source
        with patch.dict(os.environ, {"YOUTUBE_PROXY": proxy}), patch.object(youtube_source, "YoutubeDL") as ydl, \
             patch.object(youtube_source, "YouTubeTranscriptApi") as api:
            ydl.return_value.__enter__.return_value.extract_info.return_value = {"title": "T"}
            app.get_video_metadata("https://youtu.be/dQw4w9WgXcQ")
            api.return_value.list.side_effect = ValueError("stop")
            with self.assertRaises(ValueError):
                app.get_transcript("dQw4w9WgXcQ")
        self.assertEqual(ydl.call_args[0][0]["proxy"], proxy)
        config = api.call_args.kwargs["proxy_config"]
        self.assertEqual(config.to_requests_dict(), {"http": proxy, "https": proxy})


class TestLanguages(unittest.TestCase):
    def test_short_questions_follow_their_own_alphabet(self):
        self.assertEqual(app.detect_language("why?", "hi"), "en")
        self.assertEqual(app.detect_language("summary", "hi"), "en")
        self.assertEqual(app.detect_language("क्यों?", "en"), "hi")
        self.assertEqual(app.detect_language("क्यों?", "mr"), "mr")
        self.assertEqual(app.detect_language("¿por qué?", "es"), "es")
        self.assertEqual(app.detect_language("???", "hi"), "hi")

    def test_longer_questions_are_detected_the_same_way_every_time(self):
        question = "What does the speaker say about learning to stay silent?"
        self.assertEqual({app.detect_language(question, "hi") for _ in range(5)}, {"en"})
        self.assertEqual(app.detect_language("इस वीडियो में वक्ता मौन के बारे में क्या कहते हैं?", "en"), "hi")

    def test_fixed_replies_are_translated_and_answers_are_not(self):
        from grounding import INSUFFICIENT_CONTEXT, UNCERTAIN_MESSAGE, localize
        self.assertNotEqual(localize(INSUFFICIENT_CONTEXT, "hi"), INSUFFICIENT_CONTEXT)
        self.assertNotEqual(localize(UNCERTAIN_MESSAGE, "es"), UNCERTAIN_MESSAGE)
        self.assertEqual(localize("The video says: “Hola.”", "es"), "El video dice: “Hola.”")
        self.assertEqual(localize(INSUFFICIENT_CONTEXT, "zh-cn"), INSUFFICIENT_CONTEXT)
        self.assertEqual(localize("A verified answer.", "hi"), "A verified answer.")

    def pick(self, *tracks):
        items = []
        for code, generated in tracks:
            item = Mock(language_code=code, is_generated=generated)
            item.fetch.return_value.to_raw_data.return_value = [{"text": f"{code} {'auto' if generated else 'human'}", "start": 0.0}]
            items.append(item)
        import youtube_source
        with patch.object(youtube_source, "YouTubeTranscriptApi") as api:
            api.return_value.list.return_value = items
            return app.get_transcript("dQw4w9WgXcQ")[0]

    def test_regional_codes_match_and_human_captions_win(self):
        self.assertEqual(self.pick(("en", True), ("en-US", False), ("hi", False)), "en-US human")
        self.assertEqual(self.pick(("es", False), ("en-GB", True)), "en-GB auto")
        self.assertEqual(self.pick(("hi", True), ("HI", False)), "HI human")
        self.assertEqual(self.pick(("fr", True), ("fr", False)), "fr human")
        self.assertEqual(self.pick(("de", False), ("fr", False)), "de human")


class TestLanguageLabel(unittest.TestCase):
    def setUp(self):
        self.saved = dict(app.SESSION_STORE)
        app.SESSION_STORE.clear()
        self.session = app.create_session()

    def tearDown(self):
        app.SESSION_STORE.clear()
        app.SESSION_STORE.update(self.saved)

    def test_label_is_the_same_after_loading_and_after_switching_back(self):
        with patch.object(app, "get_video_metadata", return_value={"title": "T", "channel": "C"}), \
             patch.object(app, "get_transcript", return_value=("Namaste", "hi", "hi-IN", [])), \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()):
            status = app.load_video(self.session, "https://youtu.be/dQw4w9WgXcQ")[0]
        chat_id = app.SESSION_STORE[self.session]["active_chat"]
        selected = app.select_chat(self.session, chat_id)
        shown = [part for part in selected if isinstance(part, str) and "Language" in part][0]
        self.assertIn("`hi-IN`", status)
        self.assertIn("`hi-IN`", shown)
        self.assertEqual(app.SESSION_STORE[self.session]["chats"][chat_id]["language"], "hi")


class TestExportCleanup(unittest.TestCase):
    def test_old_exports_are_removed_and_new_ones_kept(self):
        import os, tempfile, time
        with tempfile.TemporaryDirectory() as directory, patch("tempfile.gettempdir", return_value=directory):
            session = app.create_session()
            chat = {"id": "c", "metadata": {"title": "Notes"}, "video_id": "dQw4w9WgXcQ", "messages": []}
            app.SESSION_STORE[session]["chats"]["c"] = chat
            try:
                first = app.export_conversation_notes(session, "c")
                old = time.time() - app.EXPORT_MAX_AGE - 60
                os.utime(os.path.dirname(first), (old, old))
                second = app.export_conversation_notes(session, "c")
                third = app.export_conversation_notes(session, "c")
            finally:
                del app.SESSION_STORE[session]
            root = os.path.join(directory, "youtube_chatbot_exports")
            self.assertFalse(os.path.exists(first))  # older than an hour, so removed
            self.assertTrue(os.path.exists(second) and os.path.exists(third))
            self.assertNotEqual(second, third)
            self.assertEqual(len(os.listdir(root)), 2)  # everything sits in one folder


class TestRequestLogging(unittest.TestCase):
    def setUp(self):
        self.saved = dict(app.SESSION_STORE)
        app.SESSION_STORE.clear()
        self.session = app.create_session()
        with patch.object(app, "get_video_metadata", return_value={"title": "T", "channel": "C"}), \
             patch.object(app, "get_transcript", return_value=("Some transcript text.", "en", "en", [])), \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()), \
             self.assertLogs("ytchat.app", level="INFO") as logs:
            app.load_video(self.session, "https://youtu.be/dQw4w9WgXcQ")
        self.load_logs = logs.output
        self.chat_id = app.SESSION_STORE[self.session]["active_chat"]

    def tearDown(self):
        app.SESSION_STORE.clear()
        app.SESSION_STORE.update(self.saved)

    def test_video_load_is_logged_with_its_time(self):
        self.assertTrue(any("video dQw4w9WgXcQ indexed in" in line for line in self.load_logs))

    def test_answer_is_logged_with_time_per_stage(self):
        chat = app.SESSION_STORE[self.session]["chats"][self.chat_id]

        def fake_answer(question, chat):
            chat["_last_retrieved_context"] = {"stage": "answered", "grounding": {"status": "verified"},
                                               "timings": {"retrieving": 0.2, "generating": 1.5}}
            return "Answer"

        with patch.object(app, "answer_question", side_effect=fake_answer), \
             self.assertLogs("ytchat.app", level="INFO") as logs:
            app.send_message(self.session, self.chat_id, "A question", [])
        line = logs.output[-1]
        self.assertIn("question answered in", line)
        self.assertIn("result verified", line)
        self.assertIn("retrieving 0.2s, generating 1.5s", line)
        self.assertIs(chat, app.SESSION_STORE[self.session]["chats"][self.chat_id])

    def test_failure_is_logged_with_stage_and_status_but_no_secret(self):
        from types import SimpleNamespace
        error = RuntimeError("token hf_secret123")
        error.response = SimpleNamespace(status_code=402)

        def failing(question, chat):
            chat["_last_retrieved_context"] = {"stage": "generating"}
            raise error

        with patch.object(app, "answer_question", side_effect=failing), \
             self.assertLogs("ytchat.app", level="ERROR") as logs:
            app.send_message(self.session, self.chat_id, "A question", [])
        text = "\n".join(logs.output)
        self.assertIn("failed at stage generating", text)
        self.assertIn("HTTP status 402", text)
        self.assertNotIn("hf_secret123", text)

    def test_real_answer_records_every_stage(self):
        chat = app.SESSION_STORE[self.session]["chats"][self.chat_id]
        chat["retriever"] = Mock(invoke=Mock(return_value=[]))
        with patch.object(app, "detect_metadata_intent", return_value=None):
            app.answer_question("Anything?", chat)
        timings = chat["_last_retrieved_context"]["timings"]
        self.assertEqual(list(timings), ["metadata", "retrieving"])


if __name__ == "__main__":
    unittest.main()
