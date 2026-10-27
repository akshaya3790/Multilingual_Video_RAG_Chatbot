"""Viewer and state tests. YouTube/model responses are fixtures, not live calls."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from langchain_core.documents import Document
import app_yt as app


class TestTranscriptViewers(unittest.TestCase):
    def setUp(self):
        self.session = app.create_session()
        self.chat = {
            "id": "first", "title": "First video",
            # One earlier exchange, so questions are follow-ups and go through the rewrite step.
            "messages": [{"role": "user", "content": "Earlier question"},
                         {"role": "assistant", "content": "Earlier answer"}],
            "metadata": {"title": "First video", "channel": "Test"},
            "language": "en", "video_id": "dQw4w9WgXcQ", "retriever": Mock(),
            "transcript_text": "First paragraph. Second paragraph.\nLine two.",
            "transcript_entries": [
                {"text": "First paragraph.", "start": 1.2},
                {"text": "Second paragraph.\nLine two.", "start": 65.8},
            ],
        }
        app.SESSION_STORE[self.session]["chats"]["first"] = self.chat

    def tearDown(self):
        app.SESSION_STORE.pop(self.session, None)

    def test_timestamped_transcript_preserves_paragraphs_and_copyable_plain_text(self):
        opened, text = app.open_transcript_viewer(self.session, "first")
        self.assertTrue(opened["open"])
        self.assertEqual(text, "[00:01] First paragraph.\n\n[01:05] Second paragraph.\nLine two.")

    def test_missing_transcript_and_invalid_session_have_clear_messages(self):
        for session, chat_id in [(None, None), ("missing", "first"), (self.session, "missing")]:
            with self.subTest(session=session):
                transcript, context = app.refresh_video_viewers(session, chat_id)
                self.assertIn("No video loaded", transcript)
                self.assertIn("No video loaded", context)
        self.assertIn("unavailable", app.transcript_view_text({"metadata": {"title": "No transcript"}}))

    def test_retained_index_text_is_preferred_when_entries_do_not_match(self):
        self.chat["transcript_entries"] = [{"text": "unrelated old entry", "start": 3}]
        self.assertEqual(app.transcript_view_text(self.chat), self.chat["transcript_text"])
        self.chat["transcript_entries"] = []
        self.assertEqual(app.transcript_view_text(self.chat), self.chat["transcript_text"])

    def test_legacy_chats_use_existing_entries_without_fetching(self):
        self.chat.pop("transcript_text")
        self.assertIn("[01:05] Second paragraph.", app.transcript_view_text(self.chat))

    def test_long_transcript_is_not_truncated_and_bad_timestamps_are_omitted(self):
        entries = [{"text": f"Segment {i} unique content.", "start": i} for i in range(6000)]
        text = app.transcript_view_text({"transcript_entries": entries})
        self.assertIn("Segment 0 unique content.", text)
        self.assertIn("Segment 5999 unique content.", text)
        self.assertEqual(text.count("unique content."), 6000)
        self.assertIn("[01:39:59]", text)
        for start in [None, "bad", float("nan"), float("inf")]:
            with self.subTest(start=start):
                self.assertEqual(app.transcript_view_text({"transcript_entries": [{"text": "Actual text", "start": start}]}), "Actual text")

    def test_load_retains_exact_index_input_and_viewers_never_repeat_io(self):
        metadata = {"title": "Loaded video", "channel": "Channel"}
        text = self.chat["transcript_text"]
        entries = self.chat["transcript_entries"]
        with patch.object(app, "get_video_metadata", return_value=metadata) as meta, \
             patch.object(app, "get_transcript", return_value=(text, "en", "en", entries)) as fetch, \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()) as index:
            result = app.load_video(self.session, "https://youtu.be/dQw4w9WgXcQ")
            chat_id = result[-1]
            loaded = app.SESSION_STORE[self.session]["chats"][chat_id]
            self.assertEqual(loaded["transcript_text"], text)
            for _ in range(3):
                app.open_transcript_viewer(self.session, chat_id)
                app.open_context_viewer(self.session, chat_id)
                app.refresh_video_viewers(self.session, chat_id)
            fetch.assert_called_once_with("dQw4w9WgXcQ")
            index.assert_called_once_with(text, "dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")
            meta.assert_called_once()

    def test_switching_videos_updates_selected_id_transcript_and_context(self):
        self.chat["_last_retrieved_context"] = {"question": "First question", "chunks": [{"text": "First context", "metadata": {}}]}
        app.SESSION_STORE[self.session]["chats"]["second"] = {
            "messages": [], "metadata": {"title": "Second video", "channel": "Test"},
            "language": "en", "transcript_text": "Second video transcript.",
        }
        _, _, selected = app.select_chat_for_viewers(self.session, "second")
        self.assertEqual(selected, "second")
        self.assertEqual(app.SESSION_STORE[self.session]["active_chat"], "second")
        transcript, context = app.refresh_video_viewers(self.session, selected)
        self.assertEqual(transcript, "Second video transcript.")
        self.assertNotIn("First context", context)
        self.assertIn("No retrieved context yet", context)
        _, _, selected = app.select_chat_for_viewers(self.session, "first")
        self.assertIn("First paragraph", app.refresh_video_viewers(self.session, selected)[0])
        self.assertIn("First context", app.refresh_video_viewers(self.session, selected)[1])

    def test_new_load_and_failed_load_do_not_display_old_context(self):
        self.chat["_last_retrieved_context"] = {"question": "Old question", "chunks": [{"text": "Old secret context", "metadata": {}}]}
        with patch.object(app, "get_video_metadata", return_value={"title": "Second", "channel": "Test"}), \
             patch.object(app, "get_transcript", return_value=("Fresh text", "en", "en", [{"text": "Fresh text", "start": 0}])), \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()):
            result = app.load_video(self.session, "https://youtu.be/jNQXAC9IVRw")
        transcript, context = app.refresh_video_viewers(self.session, result[-1])
        self.assertIn("Fresh text", transcript)
        self.assertNotIn("Old secret context", context)
        with patch.object(app, "get_video_metadata", return_value={"title": "Unavailable"}), \
             patch.object(app, "get_transcript", side_effect=ValueError("No transcript")):
            failed = app.load_video(self.session, "https://youtu.be/dQw4w9WgXcQ")
        self.assertIsNone(failed[-1])
        transcript, context = app.refresh_video_viewers(self.session, failed[-1])
        self.assertIn("No video loaded", transcript)
        self.assertNotIn("Old secret context", context)

    def test_sessions_cannot_view_each_others_transcript_or_context(self):
        other_session = app.create_session()
        try:
            self.assertIsNone(app.get_viewer_chat(other_session, "first"))
            self.assertIn("No video loaded", app.refresh_video_viewers(other_session, "first")[0])
            app.SESSION_STORE[other_session]["chats"]["first"] = {"transcript_text": "Other user's video"}
            self.assertEqual(app.refresh_video_viewers(other_session, "first")[0], "Other user's video")
            self.assertNotIn("Other user's video", app.refresh_video_viewers(self.session, "first")[0])
        finally:
            app.SESSION_STORE.pop(other_session)

    def test_context_records_exact_reranked_chunks_given_to_answer_chain(self):
        candidates = [Document(page_content="Unselected candidate"), Document(page_content="Chosen chunk", metadata={"source": "https://youtu.be/dQw4w9WgXcQ"})]
        selected = [candidates[1]]
        self.chat["retriever"].invoke.return_value = candidates
        with patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Standalone question"))), \
             patch.object(app, "rerank_documents", return_value=selected) as rerank, \
             patch.object(app, "answer_chain", Mock(invoke=Mock(return_value="Answer"))) as chain, \
             patch.object(app, "detect_metadata_intent", return_value=None):
            app.answer_question("Explain this passage", self.chat)
        self.assertEqual(chain.invoke.call_args.args[0]["context"], "Chosen chunk")
        self.chat["retriever"].invoke.assert_called_once_with("Standalone question")
        rerank.assert_called_once_with("Standalone question", candidates, top_n=4)
        saved = self.chat["_last_retrieved_context"]
        self.assertEqual(saved["chunks"], [{"text": "Chosen chunk", "metadata": selected[0].metadata}])
        display = app.retrieved_context_view_text(self.chat)
        self.assertIn("Chunk 1 (after reranking)", display)
        self.assertIn("Chosen chunk", display)
        self.assertIn("https://youtu.be/dQw4w9WgXcQ", display)
        self.assertNotIn("Unselected candidate", display)
        self.assertNotIn("score", display.lower())
        self.assertNotIn("timestamp", display.lower())

    def test_context_keeps_available_metadata_without_inventing_values(self):
        self.chat["_last_retrieved_context"] = {"question": "Question", "chunks": [{"text": "Actual chunk", "metadata": {"start": 42.5, "score": 0.72}}]}
        display = app.retrieved_context_view_text(self.chat)
        self.assertIn('"start": 42.5', display)
        self.assertIn('"score": 0.72', display)

    def test_metadata_question_and_early_failure_clear_stale_context(self):
        for failure in [False, True]:
            with self.subTest(failure=failure):
                self.chat["_last_retrieved_context"] = {"question": "Old", "chunks": [{"text": "Stale passage", "metadata": {}}]}
                if failure:
                    with patch.object(app, "detect_metadata_intent", return_value=None), \
                         patch.object(app, "rewrite_chain", Mock(invoke=Mock(side_effect=ConnectionError()))):
                        app.send_message(self.session, "first", "New question", [])
                else:
                    app.answer_question("What is the video title?", self.chat)
                display = app.retrieved_context_view_text(self.chat)
                self.assertNotIn("Stale passage", display)
                self.assertIn("No transcript chunks selected", display)

    def test_generation_failure_still_exposes_selected_context(self):
        doc = Document(page_content="Context before failed generation")
        self.chat["retriever"].invoke.return_value = [doc]
        with patch.object(app, "detect_metadata_intent", return_value=None), \
             patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Question"))), \
             patch.object(app, "rerank_documents", return_value=[doc]), \
             patch.object(app, "answer_chain", Mock(invoke=Mock(side_effect=ConnectionError()))):
            app.send_message(self.session, "first", "Explain this passage", [])
        self.assertIn(doc.page_content, app.retrieved_context_view_text(self.chat))

    def test_clear_and_new_chat_reset_display_without_deleting_transcript(self):
        self.chat["_last_retrieved_context"] = {"question": "Old", "chunks": []}
        app.clear_current_chat(self.session, "first")
        transcript, context = app.refresh_video_viewers(self.session, "first")
        self.assertIn("First paragraph", transcript)
        self.assertIn("No retrieved context yet", context)
        new_id = app.new_chat(self.session)[-1]
        self.assertIsNone(new_id)
        self.assertIn("No video loaded", app.refresh_video_viewers(self.session, new_id)[0])
        self.assertIn("First paragraph", self.chat["transcript_text"])

    def test_ui_starts_collapsed_and_refreshes_after_every_workflow(self):
        self.assertFalse(app.transcript_panel.open)
        self.assertFalse(app.context_panel.open)
        for display in [app.transcript_display, app.context_display]:
            self.assertFalse(display.interactive)
            self.assertEqual(display.lines, 14)
            self.assertEqual(display.max_lines, 14)
            self.assertIsNone(display.max_length)
            self.assertIn("copy", display.buttons)
            self.assertFalse(display.value)
        refresh_callbacks = [fn for fn in app.demo.fns.values() if fn.fn == app.refresh_video_viewers]
        self.assertEqual(len(refresh_callbacks), 10)
        select_callback = next(fn for fn in app.demo.fns.values() if fn.fn == app.select_chat_for_viewers)
        self.assertIn(app.chat_id_state, select_callback.outputs)
        self.assertEqual(app.transcript_button.value, "View Transcript")
        self.assertEqual(app.context_button.value, "View Retrieved Context")

    def test_exact_context_string_and_chunk_order_match_generation_input(self):
        docs = [Document(page_content="  First\nparagraph\n", metadata={}),
                Document(page_content="Second\n\nparagraph  ", metadata={"video_id": self.chat["video_id"]})]
        self.chat["retriever"].invoke.return_value = list(reversed(docs))
        with patch.object(app, "detect_metadata_intent", return_value=None), \
             patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Rewritten query"))) as rewrite, \
             patch.object(app, "rerank_documents", return_value=docs) as rerank, \
             patch.object(app, "answer_chain", Mock(invoke=Mock(return_value="Answer"))) as answer:
            app.answer_question("Exact user question", self.chat)
            display = app.open_context_viewer(self.session, "first")[1]
            for _ in range(2):
                app.refresh_video_viewers(self.session, "first")
            actual = answer.invoke.call_args.args[0]["context"]
            self.assertEqual(actual, "\n\n".join(doc.page_content for doc in docs))
            self.assertEqual(self.chat["_last_retrieved_context"]["formatted_context"], actual)
            self.assertEqual(display.split("Exact formatted context (context argument):\n", 1)[1], actual)
            self.assertLess(display.index("Chunk 1"), display.index("Chunk 2"))
            self.assertEqual(self.chat["_last_retrieved_context"]["chunks"][0]["text"], docs[0].page_content)
            self.assertIn("Question: Exact user question", display)
            self.assertIn("Retrieval question: Rewritten query", display)
            self.assertIn(self.chat["video_id"], display)
            self.chat["retriever"].invoke.assert_called_once()
            rewrite.invoke.assert_called_once()
            rerank.assert_called_once()
            answer.invoke.assert_called_once()

    def test_no_results_preserves_query_and_explains_absent_final_context(self):
        self.chat["retriever"].invoke.return_value = []
        with patch.object(app, "detect_metadata_intent", return_value=None), \
             patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Actual retrieval query"))), \
             patch.object(app, "rerank_documents") as rerank, patch.object(app, "answer_chain") as answer:
            app.answer_question("Unanswered question", self.chat)
        display = app.retrieved_context_view_text(self.chat)
        self.assertIn("no_results", display)
        self.assertIn("Actual retrieval query", display)
        self.assertIn("retriever returned no documents", display)
        self.assertIn("not assembled", display)
        rerank.assert_not_called()
        answer.invoke.assert_not_called()

    def test_retrieval_and_reranking_failure_stages_exclude_exception_messages(self):
        for stage in ["retrieving", "reranking"]:
            with self.subTest(stage=stage):
                self.chat["_last_retrieved_context"] = {"question": "Old", "chunks": [{"text": "Stale text", "metadata": {}}]}
                self.chat["retriever"].invoke.side_effect = ConnectionError("private error detail") if stage == "retrieving" else None
                self.chat["retriever"].invoke.return_value = [Document(page_content="Candidate")]
                with patch.object(app, "detect_metadata_intent", return_value=None), \
                     patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Actual query"))), \
                     patch.object(app, "rerank_documents", side_effect=RuntimeError("private error detail")), \
                     redirect_stdout(StringIO()) as terminal:
                    app.send_message(self.session, "first", "New question", [])
                display = app.retrieved_context_view_text(self.chat)
                self.assertIn(f"Failed stage: {stage}", display)
                self.assertIn("Actual query", display)
                self.assertNotIn("Stale text", display)
                self.assertNotIn("private error detail", display + terminal.getvalue())

    def test_empty_and_untimestamped_transcripts_are_explicit(self):
        self.assertIn("unavailable", app.transcript_view_text({"transcript_text": "", "transcript_entries": []}))
        self.assertEqual(app.transcript_view_text({"transcript_text": "Plain\n\nfull text"}), "Plain\n\nfull text")
        self.assertEqual(app.transcript_view_text({"transcript_entries": [{"text": "No invented timestamp", "start": -1}]}), "No invented timestamp")

    def test_manual_request_streams_pending_before_retrieval_and_preserves_arguments(self):
        self.chat["_last_retrieved_context"] = {"question": "Old", "chunks": [{"text": "Stale content", "metadata": {}}]}
        doc = Document(page_content="Fresh selected text")
        self.chat["retriever"].invoke.return_value = [doc]
        with patch.object(app, "detect_metadata_intent", return_value=None), \
             patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Query"))) as rewrite, \
             patch.object(app, "rerank_documents", return_value=[doc]), \
             patch.object(app, "answer_chain", Mock(invoke=Mock(return_value="Answer"))) as answer:
            stream = app.send_with_context(self.session, "first", "Latest question", [])
            pending = next(stream)
            self.assertIn("Latest question", pending[2])
            self.assertIn("pending", pending[2])
            self.assertNotIn("Stale content", pending[2])
            self.chat["retriever"].invoke.assert_not_called()
            rewrite.invoke.assert_not_called()
            answer.invoke.assert_not_called()
            finished = next(stream)
            self.assertEqual(finished[0][0]["content"], "Latest question")
            self.assertEqual(finished[1], "")
            self.assertEqual(finished[2], app.gr.skip())
            self.assertIn(doc.page_content, app.refresh_video_viewers(self.session, "first")[1])
            answer.invoke.assert_called_once()
            with self.assertRaises(StopIteration):
                next(stream)

    def test_each_suggestion_streams_pending_and_updates_its_actual_question(self):
        handlers = [app.send_suggestion_1, app.send_suggestion_2, app.send_suggestion_3, app.send_suggestion_4]
        doc = Document(page_content="Selected evidence")
        self.chat["retriever"].invoke.return_value = [doc]
        for handler in handlers:
            with self.subTest(handler=handler.__name__), \
                 patch.object(app, "detect_metadata_intent", return_value=None), \
                 patch.object(app, "rewrite_chain", Mock(invoke=Mock(side_effect=lambda inputs: inputs["question"]))), \
                 patch.object(app, "rerank_documents", return_value=[doc]), \
                 patch.object(app, "answer_chain", Mock(invoke=Mock(return_value="Answer"))):
                callback = next(fn.fn for fn in app.demo.fns.values() if getattr(fn.fn, "__wrapped__", None) is handler)
                stream = callback(self.session, "first", [])
                self.assertIn("pending", next(stream)[2])
                finished = next(stream)
                actual_question = finished[0][0]["content"]
                context = app.refresh_video_viewers(self.session, "first")[1]
                self.assertIn("Question: " + actual_question, context)
                self.assertIn("Retrieval question: " + actual_question, context)
                self.assertIn(doc.page_content, context)

    def test_refresh_dependencies_cover_send_enter_suggestions_and_video_workflows(self):
        expected = [app.load_event, app.send_event, app.submit_event, app.select_event,
                    app.new_event, app.clear_event, *app.suggestion_events]
        dependencies = app.demo.config["dependencies"]
        for event in expected:
            matches = [d for d in dependencies if d.get("trigger_after") == event["id"]]
            self.assertTrue(any(app.demo.fns[d["id"]].fn is app.refresh_video_viewers for d in matches))
        for event in [app.send_event, app.submit_event]:
            callback = app.demo.fns[event["id"]]
            self.assertIs(callback.fn, app.send_with_context)
            self.assertIs(callback.fn.__wrapped__, app.send_message)
            self.assertEqual(callback.outputs, [app.chatbot, app.message, app.context_display])
        self.assertIs(app.demo.fns[app.load_event["id"]].fn.__wrapped__, app.load_video)

    def test_loading_stream_hides_old_video_until_actual_processing_finishes(self):
        with patch.object(app, "get_video_metadata", return_value={"title": "New video", "channel": "Test"}) as metadata, \
             patch.object(app, "get_transcript", return_value=("New full transcript", "en", "en", [])) as fetch, \
             patch.object(app, "build_hybrid_retriever", return_value=Mock()) as index:
            stream = app.load_video_with_viewers(self.session, "https://youtu.be/jNQXAC9IVRw")
            pending = next(stream)
            self.assertIn("processing is in progress", pending[-2])
            self.assertNotIn("First paragraph", app.open_transcript_viewer(self.session, "first")[1])
            metadata.assert_not_called()
            fetch.assert_not_called()
            index.assert_not_called()
            finished = next(stream)
            self.assertEqual(finished[-2], "New full transcript")
            self.assertNotIn("First paragraph", finished[-2])
            self.assertFalse(app.SESSION_STORE[self.session].get("_viewer_loading"))
            fetch.assert_called_once()
            index.assert_called_once()
        stream = app.load_video_with_viewers(self.session, "unused")
        next(stream)
        stream.close()
        self.assertNotIn("_viewer_loading", app.SESSION_STORE[self.session])

    def test_generation_failure_preserves_exact_context_and_failure_stage(self):
        doc = Document(page_content="Preserved context\nwith exact line breaks")
        self.chat["retriever"].invoke.return_value = [doc]
        with patch.object(app, "detect_metadata_intent", return_value=None), \
             patch.object(app, "rewrite_chain", Mock(invoke=Mock(return_value="Query"))), \
             patch.object(app, "rerank_documents", return_value=[doc]), \
             patch.object(app, "answer_chain", Mock(invoke=Mock(side_effect=ConnectionError("not exposed")))):
            app.send_message(self.session, "first", "Question", [])
        saved = self.chat["_last_retrieved_context"]
        self.assertEqual(saved["formatted_context"], doc.page_content)
        self.assertEqual(saved["failed_stage"], "generating")
        display = app.retrieved_context_view_text(self.chat)
        self.assertEqual(display.split("Exact formatted context (context argument):\n", 1)[1], doc.page_content)
        self.assertNotIn("not exposed", display)

    def test_metadata_route_explicitly_marks_query_unavailable(self):
        with patch.object(app, "rewrite_chain") as rewrite, patch.object(app, "answer_chain") as answer:
            app.answer_question("What is the video title?", self.chat)
        display = app.retrieved_context_view_text(self.chat)
        self.assertIn("metadata_only", display)
        self.assertIn("Retrieval question: unavailable", display)
        self.assertIn("bypasses retrieval", display)
        rewrite.invoke.assert_not_called()
        answer.invoke.assert_not_called()

    def test_older_request_does_not_overwrite_a_newer_context_record(self):
        newer = {"question": "Newer request", "chunks": [], "stage": "pending"}
        def newer_request_started(inputs):
            self.chat["_last_retrieved_context"] = newer
            return "Older retrieval query"
        self.chat["retriever"].invoke.return_value = [Document(page_content="Older context")]
        with patch.object(app, "detect_metadata_intent", return_value=None), \
             patch.object(app, "rewrite_chain", Mock(invoke=Mock(side_effect=newer_request_started))), \
             patch.object(app, "rerank_documents", return_value=[Document(page_content="Older context")]), \
             patch.object(app, "answer_chain", Mock(invoke=Mock(return_value="Older answer"))):
            app.answer_question("Older question", self.chat)
        self.assertIs(self.chat["_last_retrieved_context"], newer)
        self.assertNotIn("Older context", app.retrieved_context_view_text(self.chat))

    def test_first_question_skips_the_rewrite_call(self):
        self.chat["messages"] = []
        self.chat["retriever"].invoke.return_value = []
        with patch.object(app, "detect_metadata_intent", return_value=None), \
             patch.object(app, "rewrite_chain") as rewrite:
            app.answer_question("Opening question", self.chat)
        rewrite.invoke.assert_not_called()
        self.chat["retriever"].invoke.assert_called_once_with("Opening question")


if __name__ == "__main__":
    unittest.main()
