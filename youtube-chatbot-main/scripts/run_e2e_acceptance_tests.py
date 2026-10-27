import os
import sys
import json
from unittest.mock import patch

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# These manual runners live in scripts/; the app modules are one folder up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app_yt

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)

def run_e2e_acceptance():
    print_banner("STARTING FINAL END-TO-END ACCEPTANCE TESTS")

    # =========================================================================
    # TEST 1: LOAD A REAL YOUTUBE VIDEO (LIVE INTEGRATION)
    # =========================================================================
    print_banner("TEST 1: LOAD REAL YOUTUBE VIDEO (LIVE FETCH FROM YOUTUBE)")
    session_id = app_yt.create_session()
    real_url = "https://youtu.be/3jV9hwVLEls?si=6uep3W3NupMSEdI-"
    print(f"Loading real video URL: {real_url}")

    status_out, choices_out, bot_out, msg_out, chat_id = app_yt.load_video(session_id, real_url)

    print("\n--- Live Video Load Output ---")
    print(f"Status Output:\n{status_out}")
    print(f"Chat Choices: {choices_out.get('choices')}")
    print(f"Chatbot Output: {bot_out}")
    print(f"Active Chat ID: {chat_id}")

    assert chat_id is not None, "Live video load failed to produce chat_id"
    chat = app_yt.SESSION_STORE[session_id]["chats"][chat_id]
    metadata = chat["metadata"]
    transcript_entries = chat["transcript_entries"]

    print("\n--- Extracted Live Video Metadata ---")
    print(f"Title: {metadata.get('title')}")
    print(f"Channel: {metadata.get('channel')}")
    print(f"Duration: {metadata.get('duration')}s")
    print(f"View Count: {metadata.get('view_count')}")
    print(f"Live Transcript Entries Count: {len(transcript_entries)}")
    assert len(transcript_entries) > 0, "Live transcript entries must be non-empty"
    assert "start" in transcript_entries[0], "Transcript entries must have 'start' timestamp"

    # Ask a live metadata question (handled locally by metadata router without external LLM)
    meta_q = "What is the title of this video?"
    bot_out, _ = app_yt.send_message(session_id, chat_id, meta_q, [])
    print(f"\nQuestion: {meta_q}")
    print(f"Answer: {bot_out[-1]['content']}")
    assert metadata["title"] in bot_out[-1]["content"], "Metadata title must be in answer"
    print(">>> TEST 1 RESULT: PASS (Live YouTube metadata and transcript fetched successfully)")

    # =========================================================================
    # TEST 2: VERIFY TIMESTAMP REFERENCES (LIVE ENTRIES + CONTROLLED MODEL)
    # =========================================================================
    print_banner("TEST 2: VERIFY TIMESTAMP REFERENCES")
    # Mock answer generation and support decisions to test the citation resolver
    # against the REAL transcript entries retrieved live from YouTube.
    sample_entry = next(entry for entry in transcript_entries if len(entry['text'].split()) >= 4)
    moment_text = sample_entry["text"]
    moment_start = int(float(sample_entry["start"]))
    m, s = divmod(moment_start, 60)
    expected_timestamp = f"{m:02d}:{s:02d}"
    expected_url = f"https://www.youtube.com/watch?v={chat['video_id']}&t={moment_start}s"

    print(f"Target real transcript segment: '{moment_text}' at {expected_timestamp} ({moment_start}s)")
    simulated_answer = moment_text

    def mock_invoke_answer(inputs, *args, **kwargs):
        if isinstance(inputs, dict) and 'data' in inputs:
            return json.dumps({'supported': True, 'answers_question': True, 'evidence_indices': [0]})
        if isinstance(inputs, dict) and "answer_language" in inputs:
            return json.dumps({'answer': simulated_answer, 'evidence': [moment_text]})
        return "What statement is made in this passage?"

    with patch.object(app_yt.answer_chain.__class__, "invoke", side_effect=mock_invoke_answer):
        with patch("app_yt.rerank_documents", return_value=[app_yt.Document(page_content=f"Intro: {moment_text} and background music.", metadata={})]):
            bot_out, _ = app_yt.send_message(session_id, chat_id, "What statement is made in this passage?", bot_out)

    assistant_response = bot_out[-1]["content"]
    print(f"\nAssistant Response with Source References:\n{assistant_response}")

    # Verifications:
    assert simulated_answer in assistant_response, "Original answer text must be preserved verbatim"
    assert "<details open class='source-references-box'>" in assistant_response, "Collapsible details box must be present"
    assert expected_timestamp in assistant_response, f"Timestamp {expected_timestamp} must be present"
    assert expected_url in assistant_response, f"URL {expected_url} must be present"
    print(">>> Verified: Referenced timestamp corresponds to exact source moment in video.")
    print(">>> Verified: Link opens video at exact start seconds.")

    # Test case where no reliable timestamp can be established (unrelated passage):
    def mock_invoke_unrelated(inputs, *args, **kwargs):
        if isinstance(inputs, dict) and 'data' in inputs:
            return json.dumps({'supported': False, 'answers_question': False})
        if isinstance(inputs, dict) and "answer_language" in inputs:
            return json.dumps({'answer': 'General knowledge without specific transcript passage.',
                               'evidence': ['completely unrelated string xyz']})
        return "Explain quantum physics"

    with patch.object(app_yt.answer_chain.__class__, "invoke", side_effect=mock_invoke_unrelated):
        with patch("app_yt.rerank_documents", return_value=[app_yt.Document(page_content="completely unrelated string xyz", metadata={})]):
            bot_out, _ = app_yt.send_message(session_id, chat_id, "Explain quantum physics", bot_out)

    clean_ans = bot_out[-1]["content"]
    print(f"\nResponse without matching passage:\n{clean_ans}")
    assert "<details" not in clean_ans, "Must NOT fabricate a citation when no passage matches"
    assert clean_ans == "The available video context does not provide enough information to answer this question."
    print(">>> TEST 2 RESULT: PASS (Real transcript timestamps linked; ungrounded answers produce 0 citations)")

    # =========================================================================
    # TEST 3: VERIFY REAL CONVERSATION EXPORT
    # =========================================================================
    print_banner("TEST 3: VERIFY REAL CONVERSATION EXPORT")
    msg_count_before = len(chat["messages"])
    active_chat_before = app_yt.SESSION_STORE[session_id]["active_chat"]
    print(f"Message count before export: {msg_count_before}")
    print(f"Active chat before export: {active_chat_before}")

    exported_file = app_yt.export_conversation_notes(session_id, chat_id)
    print(f"\nExported file path: {exported_file}")
    assert exported_file and os.path.exists(exported_file), "Export file must exist"
    assert exported_file.endswith(".md"), "Export file must be a Markdown (.md) file"

    msg_count_after = len(chat["messages"])
    active_chat_after = app_yt.SESSION_STORE[session_id]["active_chat"]
    print(f"Message count after export: {msg_count_after}")
    print(f"Active chat after export: {active_chat_after}")
    assert msg_count_before == msg_count_after, "Exporting must NOT alter message count"
    assert active_chat_before == active_chat_after, "Exporting must NOT alter active chat"

    with open(exported_file, "r", encoding="utf-8") as f:
        notes_content = f.read()

    print("\n--- Exported Notes File Content ---")
    print(notes_content)

    # Verifications:
    assert "# YouTube Chatbot — Conversation Notes" in notes_content
    assert metadata["title"] in notes_content
    assert f"https://www.youtube.com/watch?v={chat['video_id']}" in notes_content
    assert "### User" in notes_content
    assert "What is the title of this video?" in notes_content
    assert "### Assistant" in notes_content
    assert simulated_answer in notes_content
    assert "## Source References" in notes_content
    assert expected_url in notes_content
    print(">>> TEST 3 RESULT: PASS (Export populated with real conversation dialogue and source references)")

    # =========================================================================
    # TEST 4: VERIFY SUGGESTED QUESTIONS
    # =========================================================================
    print_banner("TEST 4: VERIFY SUGGESTED QUESTIONS")
    current_bot = bot_out

    with patch("app_yt.answer_question", side_effect=lambda q, c: f"Answer to: {q}"):
        # Suggestion 1
        current_bot, _ = app_yt.send_suggestion_1(session_id, chat_id, current_bot)
        assert current_bot[-2]["content"] == "What are the main concepts discussed in this video?"
        assert current_bot[-1]["content"] == "Answer to: What are the main concepts discussed in this video?"

        # Suggestion 2
        current_bot, _ = app_yt.send_suggestion_2(session_id, chat_id, current_bot)
        assert current_bot[-2]["content"] == "Explain the key concepts of this video in simple terms."

        # Suggestion 3
        current_bot, _ = app_yt.send_suggestion_3(session_id, chat_id, current_bot)
        assert current_bot[-2]["content"] == "What examples does the speaker provide in this video?"

        # Suggestion 4
        current_bot, _ = app_yt.send_suggestion_4(session_id, chat_id, current_bot)
        assert current_bot[-2]["content"] == "What are the key takeaways from this video?"

    print("All 4 suggested questions submitted successfully through send_message.")

    # Manual Question Submission
    with patch("app_yt.answer_question", return_value="Manual answer to custom question."):
        current_bot, msg_out = app_yt.send_message(session_id, chat_id, "What genre is this music video?", current_bot)
        assert msg_out == ""
        assert current_bot[-2]["content"] == "What genre is this music video?"
        assert current_bot[-1]["content"] == "Manual answer to custom question."

    print("Manual question submission verified intact.")
    print(">>> TEST 4 RESULT: PASS (All 4 suggestions and manual submission function cleanly)")

    # =========================================================================
    # TEST 5: RECHECK EXISTING WORKFLOWS
    # =========================================================================
    print_banner("TEST 5: RECHECK EXISTING WORKFLOWS")
    # 1. New chat
    new_bot, new_status, new_chat_id = app_yt.new_chat(session_id)
    assert new_bot == []
    assert new_chat_id is None
    assert "New conversation" in new_status
    print("New Chat verified.")

    # 2. Select previous chat
    restored_bot, restored_status = app_yt.select_chat(session_id, chat_id)
    assert len(restored_bot) > 0
    assert metadata["title"] in restored_status
    print("Conversation switching verified.")

    # 3. Clear current chat
    cleared_bot = app_yt.clear_current_chat(session_id, chat_id)
    assert cleared_bot == []
    assert len(app_yt.SESSION_STORE[session_id]["chats"][chat_id]["messages"]) == 0
    print("Clear Current Chat verified.")

    # 4. Load a second video (verifying no stale state)
    second_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    status2, choices2, bot2, msg2, chat_id2 = app_yt.load_video(session_id, second_url)
    assert chat_id2 != chat_id
    chat2 = app_yt.SESSION_STORE[session_id]["chats"][chat_id2]
    print(f"Second video loaded: '{chat2['metadata']['title']}' (ID: {chat2['video_id']})")
    assert chat2["video_id"] == "jNQXAC9IVRw"
    assert chat2["metadata"]["title"] != metadata["title"]
    print("Second video loaded cleanly without stale references or metadata.")
    print(">>> TEST 5 RESULT: PASS (All existing workflows preserved and functional)")

    print_banner("ALL END-TO-END ACCEPTANCE TESTS PASSED (0 FAILURES, 0 ERRORS)")

if __name__ == "__main__":
    run_e2e_acceptance()
