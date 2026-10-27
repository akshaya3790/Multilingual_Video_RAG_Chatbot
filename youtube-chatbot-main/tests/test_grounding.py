"""Grounding workflow regressions. Providers are fixtures, not live model tests."""
import json
import unittest
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from test_llm_providers import mocked_app


FALLBACK = 'The available video context does not provide enough information to answer this question.'
QUOTE = 'Silence is about moving from compulsiveness to consciousness.'


class TestGroundingWorkflow(unittest.TestCase):
    def answer(self, draft, *, supported=True, relevant=True, entries=None, documents=None, indices=None, replies=None, language=None):
        app, *_ = mocked_app({'LLM_PROVIDER': 'ollama'})
        if language:
            app['detect_language'] = Mock(return_value=language)
        entries = entries if entries is not None else [
            {'text': 'You can even speak in silence.', 'start': 12},
            {'text': QUOTE, 'start': 189.8},
        ]
        context = ' '.join(e['text'] for e in entries)
        documents = documents if documents is not None else [Document(page_content=context)]
        chat = dict(messages=[], metadata={'title': 'Silence', 'description': 'Untrusted metadata claim'},
                    language='en', retriever=Mock(), video_id='3jV9hwVLEls',
                    transcript_entries=entries, transcript_text=context)
        chat['retriever'].invoke.return_value = documents
        app['rerank_documents'] = Mock(return_value=documents)
        app['answer_chain'] = Mock(invoke=Mock(return_value=draft))
        decisions = []

        def verify(prompt):
            decisions.append(prompt.to_string())
            if replies is not None:
                return AIMessage(content=json.dumps(replies.pop(0)))
            return AIMessage(content=json.dumps({'supported': supported, 'answers_question': relevant,
                                                'evidence_indices': [0] if indices is None else indices}))

        app['llm'] = RunnableLambda(verify)
        result = app['answer_question']('What does silence mean in this video?', chat)
        return result, chat, decisions, app

    def test_general_knowledge_is_rejected_without_citations(self):
        result, chat, _, _ = self.answer('The boiling point of liquid nitrogen is -196°C.')
        self.assertEqual(result, FALLBACK)
        self.assertEqual(chat['_last_references'], [])

    def test_fallback_reply_is_in_the_language_of_the_question(self):
        from grounding import TRANSLATIONS
        result, _, _, _ = self.answer('The boiling point of liquid nitrogen is -196°C.', language='hi')
        self.assertEqual(result, TRANSLATIONS[FALLBACK]['hi'])
        result, _, _, _ = self.answer('The boiling point of liquid nitrogen is -196°C.', language='ja')
        self.assertEqual(result, FALLBACK)  # no translation yet, so English

    def test_abbreviations_and_initials_do_not_create_extra_claims(self):
        from grounding import split_sentences
        self.assertEqual(split_sentences('Dr. Jain explains it, e.g. with breathing. Mr. J. K. Rao agrees.'),
                         ['Dr. Jain explains it, e.g. with breathing.', 'Mr. J. K. Rao agrees.'])
        self.assertEqual(split_sentences('पहला वाक्य। दूसरा वाक्य।'), ['पहला वाक्य।', 'दूसरा वाक्य।'])
        self.assertEqual(split_sentences('It costs 3.5 dollars. Ok?'), ['It costs 3.5 dollars.', 'Ok?'])
        _, _, checks, _ = self.answer(json.dumps({'answer': 'Dr. Jain says: ' + QUOTE, 'evidence': [QUOTE]}))
        self.assertEqual(len(checks), 1)  # one claim, not "Dr." plus the rest

    def test_model_added_links_are_removed_instead_of_discarding_the_answer(self):
        draft = QUOTE + ' See [the clip](https://evil.example/x) or https://evil.example/y.'
        result, chat, checks, _ = self.answer(json.dumps({'answer': draft, 'evidence': [QUOTE]}))
        shown = result.split('<details')[0]
        self.assertIn(QUOTE, shown)
        self.assertIn('See the clip or', shown)
        self.assertNotIn('evil.example', result)
        self.assertNotIn('evil.example', ''.join(checks))
        self.assertEqual(chat['_last_references'][0]['seconds'], 189)

    def test_supported_structured_answer_is_rendered_without_internal_json(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': QUOTE, 'evidence': [QUOTE]}))
        self.assertEqual(result.split('<details')[0].strip(), QUOTE)
        self.assertNotIn('"evidence"', result)
        self.assertEqual(len(checks), 1)
        self.assertEqual(chat['_last_references'][0]['seconds'], 189)

    def test_citation_targets_supporting_quote_not_first_retrieved_fragment(self):
        result, chat, _, _ = self.answer(json.dumps({'answer': QUOTE, 'evidence': [QUOTE]}))
        self.assertIn('3jV9hwVLEls&t=189s', result)
        self.assertNotIn('t=12s', result)
        self.assertEqual(chat['_last_references'][0]['excerpt'], QUOTE)

    def test_invented_quote_is_rejected_before_semantic_verification(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': 'Nitrogen boils at -196°C.',
                    'evidence': ['The boiling point of nitrogen is -196°C.']}))
        self.assertEqual(result, FALLBACK)
        self.assertEqual(checks, [])
        self.assertEqual(chat['_last_references'], [])

    def test_valid_quote_does_not_license_an_unsupported_claim(self):
        result, chat, _, _ = self.answer(json.dumps({'answer': 'Silence cures every illness.',
                    'evidence': [QUOTE]}), supported=False)
        self.assertEqual(result, FALLBACK)
        self.assertEqual(chat['_last_references'], [])

    def test_generic_repeated_answer_that_does_not_answer_question_is_rejected(self):
        result, chat, _, _ = self.answer(json.dumps({'answer': QUOTE, 'evidence': [QUOTE]}), relevant=False)
        self.assertEqual(result, FALLBACK)
        self.assertEqual(chat['_last_references'], [])

    def test_quote_must_come_from_context_not_just_elsewhere_in_transcript(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': QUOTE, 'evidence': [QUOTE]}),
            documents=[Document(page_content='Only a different passage was retrieved.')])
        self.assertEqual(result, FALLBACK)
        self.assertEqual(chat['_last_references'], [])
        self.assertFalse(any('You check an answer against transcript quotations' in check for check in checks))

    def test_literal_supported_plain_string_preserves_existing_invocation_contract(self):
        result, chat, _, _ = self.answer(QUOTE)
        self.assertIn(QUOTE, result)
        self.assertIn('t=189s', result)
        self.assertEqual(chat['_last_references'][0]['seconds'], 189)

    def test_plain_quote_must_still_address_the_question(self):
        result, chat, _, _ = self.answer(QUOTE, relevant=False)
        self.assertEqual(result, FALLBACK)
        self.assertEqual(chat['_last_references'], [])

    def test_inexact_quote_is_discarded_before_verifying_remaining_evidence(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': QUOTE,
                    'evidence': [QUOTE, 'This quotation is completely invented.']}))
        self.assertIn(QUOTE, result)
        self.assertNotIn('completely invented', checks[0])
        self.assertEqual(chat['_last_references'][0]['excerpt'], QUOTE)

    def test_unknown_number_is_rejected_even_if_a_verifier_would_approve(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': 'Silence takes 196 days.',
                    'evidence': [QUOTE]}))
        self.assertEqual(result, FALLBACK)
        self.assertEqual(checks, [])
        self.assertEqual(chat['_last_references'], [])

    def test_only_exact_source_span_is_kept_from_an_expanded_quote(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': QUOTE,
                    'evidence': ['Invented introduction. ' + QUOTE + ' Invented ending.']}))
        self.assertIn(QUOTE, result)
        self.assertEqual(chat['_last_references'][0]['excerpt'], QUOTE)
        self.assertNotIn('Invented', checks[0])

    def test_unused_quote_is_not_cited_just_because_it_was_retrieved(self):
        result, chat, _, _ = self.answer(json.dumps({'answer': QUOTE,
                    'evidence': ['You can even speak in silence.', QUOTE]}), indices=[1])
        self.assertIn('t=189s', result)
        self.assertNotIn('t=12s', result)
        self.assertEqual(len(chat['_last_references']), 1)

    def test_invalid_verifier_evidence_index_is_rejected(self):
        result, chat, _, _ = self.answer(json.dumps({'answer': QUOTE, 'evidence': [QUOTE]}), indices=[99])
        self.assertEqual(result, FALLBACK)
        self.assertEqual(chat['_last_references'], [])

    def test_rejected_prose_can_only_be_repaired_with_verified_literal_source_text(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': 'Silence cures every illness.',
                'evidence': [QUOTE]}), entries=[{'text': QUOTE, 'start': 189.8}], replies=[
            {'supported': False, 'answers_question': False}, {'index': 0},
            {'supported': True, 'answers_question': True},
            {'answer': 'Silence cures everything.'}, {'supported': False, 'answers_question': True}])
        shown = result.split('<details')[0]
        self.assertIn('The video says: “' + QUOTE + '”', shown)  # introduced, not a bare fragment
        self.assertNotIn('cures', result)
        self.assertEqual(len(checks), 5)
        self.assertEqual(chat['_last_references'][0]['seconds'], 189)

    def test_repaired_answer_is_phrased_as_a_sentence_when_the_wording_is_verified(self):
        answer = 'In this video, silence means moving from compulsiveness to consciousness.'
        result, chat, checks, _ = self.answer(json.dumps({'answer': 'Silence cures every illness.',
                'evidence': [QUOTE]}), entries=[{'text': QUOTE, 'start': 189.8}], replies=[
            {'supported': False, 'answers_question': False}, {'index': 0},
            {'supported': True, 'answers_question': True},
            {'answer': answer}, {'supported': True, 'answers_question': True}])
        self.assertEqual(result.split('<details')[0].strip(), answer)
        self.assertEqual(chat['_last_references'][0]['excerpt'], QUOTE)
        self.assertIn(QUOTE, checks[-1])  # the wording was checked against the quote

    def test_rephrased_answer_cannot_add_numbers_or_links(self):
        for answer in ['Silence takes 40 days.', 'See [this](https://evil.example).']:
            result, _, checks, _ = self.answer(json.dumps({'answer': 'Silence cures every illness.',
                    'evidence': [QUOTE]}), entries=[{'text': QUOTE, 'start': 189.8}], replies=[
                {'supported': False, 'answers_question': False}, {'index': 0},
                {'supported': True, 'answers_question': True},
                {'answer': answer}, {'supported': True, 'answers_question': True}])
            self.assertIn('The video says: “' + QUOTE + '”', result)
            self.assertEqual(len(checks), 4)  # rejected before it reached the verifier

    def test_quote_repair_cannot_invent_a_nonexistent_index(self):
        result, chat, checks, _ = self.answer(json.dumps({'answer': 'An unsupported claim.',
                'evidence': [QUOTE]}), replies=[
            {'supported': False, 'answers_question': False}, {'index': 99}])
        self.assertEqual(result, FALLBACK)
        self.assertEqual(chat['_last_references'], [])
        self.assertEqual(len(checks), 2)

    def test_sentence_checks_run_concurrently_and_any_failure_still_rejects(self):
        import threading, time
        from grounding import ground_answer
        running, peak, lock = [0], [0], threading.Lock()

        def verifier(fail_marker):
            def verify(prompt):
                with lock:
                    running[0] += 1
                    peak[0] = max(peak[0], running[0])
                time.sleep(0.3)
                with lock:
                    running[0] -= 1
                text = prompt.to_string()
                ok = fail_marker not in text
                return AIMessage(content=json.dumps({'supported': ok, 'answers_question': ok, 'evidence_indices': [0]}))
            return RunnableLambda(verify)

        answer = 'Silence is about consciousness. It moves us from compulsiveness. It is about moving.'
        draft = json.dumps({'answer': answer, 'evidence': [QUOTE]})
        started = time.monotonic()
        result = ground_answer(draft, QUOTE, 'What does silence mean?', verifier('NO_FAILURE'))
        self.assertLess(time.monotonic() - started, 0.8)  # three 0.3s checks, not run one by one
        self.assertEqual(peak[0], 3)
        self.assertEqual(result.status, 'verified')

        result = ground_answer(draft, QUOTE, 'What does silence mean?', verifier('It moves us'))
        self.assertNotEqual(result.status, 'verified')


class TestEvidenceReferences(unittest.TestCase):
    def test_repair_catalogue_does_not_duplicate_context_or_exceed_its_budget(self):
        from grounding import _quotation_candidates
        context = ' '.join(f'This is source sentence number {number}.' for number in range(60))
        candidates = _quotation_candidates(context)
        self.assertLessEqual(len(candidates), 64)
        self.assertLessEqual(sum(map(len, candidates)), len(context))
        self.assertTrue(all(quote in context for quote in candidates))
        self.assertIn('number 59.', candidates[-1])

    def test_full_quote_can_span_timestamped_entries(self):
        from grounding import source_references_for_evidence
        entries = [{'text': 'Silence is about', 'start': 8.9},
                   {'text': 'moving from compulsiveness to', 'start': 11.1},
                   {'text': 'consciousness.', 'start': 12.2}]
        refs = source_references_for_evidence([QUOTE], entries, '3jV9hwVLEls')
        self.assertEqual(refs[0]['seconds'], 8)
        self.assertEqual(refs[0]['excerpt'], QUOTE)

    def test_unavailable_or_invalid_timestamps_never_become_zero_citations(self):
        from grounding import source_references_for_evidence
        for start in [None, 'bad', -1, float('inf'), float('nan')]:
            with self.subTest(start=start):
                self.assertEqual(source_references_for_evidence([QUOTE],
                    [{'text': QUOTE, 'start': start}], '3jV9hwVLEls'), [])

    def test_fragment_from_inside_a_word_is_not_transcript_evidence(self):
        from grounding import ground_answer
        result = ground_answer('He means silence is about consciousness.',
            'He means silence is about consciousnessness.', 'What does he mean?',
            RunnableLambda(lambda _: AIMessage(content='{"index": null}')))
        self.assertEqual(result.text, FALLBACK)

    def test_model_links_cannot_bypass_application_citation_checks(self):
        from grounding import ground_answer
        approve = RunnableLambda(lambda prompt: AIMessage(content=json.dumps(
            {'supported': True, 'answers_question': True, 'evidence_indices': [0]})))
        for answer in ['See https://evil.example/', 'Watch [here](https://evil.example/)',
                       '<a href="https://evil.example/">here</a>']:
            result = ground_answer(json.dumps({'answer': answer, 'evidence': [QUOTE]}),
                QUOTE, 'What does silence mean?', approve)
            # The link is removed; it never reaches the page even if the checker approves.
            self.assertNotIn('evil.example', result.text)
            self.assertNotIn('](', result.text)


if __name__ == '__main__':
    unittest.main()
