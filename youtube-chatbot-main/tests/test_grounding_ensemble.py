"""Mocked ensemble orchestration tests; actual model results are logged separately."""
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from grounding import ground_answer, source_references_for_evidence
from grounding_ensemble import (Outcome, TECH_ERROR_MESSAGE, UNCERTAIN_MESSAGE,
                               combine_support, ground_with_ensemble)
from factual_support import SupportClassifierError


QUOTE = 'Silence is about moving from compulsiveness to consciousness.'


class TestGroundingEnsemble(unittest.TestCase):
    def check(self, answer=QUOTE, evidence=None, relevant=True, checker=None, reply=None):
        calls = []
        def respond(prompt):
            calls.append(prompt.to_string())
            return AIMessage(content=reply if reply is not None else json.dumps({'answers_question': relevant}))
        with patch('grounding_ensemble.ensemble.check', checker or Mock(return_value='supported')):
            result = ground_with_ensemble(json.dumps(dict(answer=answer, evidence=evidence or [QUOTE])),
                QUOTE, 'What does silence mean?', RunnableLambda(respond))
        return result, calls

    def test_supported_answer_and_existing_citation_mapper(self):
        result, calls = self.check()
        self.assertEqual(result.status, Outcome.SUPPORTED)
        self.assertEqual(result.text, QUOTE)
        self.assertEqual(len(calls), 1)
        refs = source_references_for_evidence(result.evidence, [{'text': QUOTE, 'start': 189.8}], '3jV9hwVLEls')
        self.assertEqual(refs[0]['link'], 'https://www.youtube.com/watch?v=3jV9hwVLEls&t=189s')

    def test_support_is_separate_from_whole_answer_relevance(self):
        result, calls = self.check(relevant=False)
        self.assertEqual(result.status, Outcome.UNCERTAIN)
        self.assertEqual(result.evidence, [])
        self.assertNotIn('insufficient', result.text)

    def test_unsupported_claim_is_removed_and_has_no_reference(self):
        result, calls = self.check(answer='Silence cures illness.', checker=Mock(return_value='unsupported'))
        self.assertEqual(result.status, Outcome.UNSUPPORTED_REMOVED)
        self.assertEqual(result.evidence, [])
        self.assertEqual(calls, [])

    def test_uncertainty_is_not_missing_context(self):
        result, _ = self.check(checker=Mock(return_value='uncertain'))
        self.assertEqual(result.status, Outcome.UNCERTAIN)
        self.assertEqual(result.text, UNCERTAIN_MESSAGE)

    def test_checker_error_is_not_missing_context_or_secret_leak(self):
        result, _ = self.check(checker=Mock(side_effect=RuntimeError('secret test marker')))
        self.assertEqual(result.status, Outcome.TECH_ERROR)
        self.assertEqual(result.text, TECH_ERROR_MESSAGE)
        self.assertNotIn('secret test marker', result.text)

    def test_invalid_relevance_json_is_a_technical_error(self):
        result, _ = self.check(reply='not JSON')
        self.assertEqual(result.status, Outcome.TECH_ERROR)

    def test_malformed_draft_is_a_technical_error(self):
        result = ground_with_ensemble('{broken', QUOTE, 'question', Mock())
        self.assertEqual(result.status, Outcome.TECH_ERROR)

    def test_false_or_stitched_quote_cannot_reach_models(self):
        checker = Mock(return_value='supported')
        result, _ = self.check(evidence=['Silence is about moving from compulsiveness to dancing.'], checker=checker)
        self.assertEqual(result.status, Outcome.UNSUPPORTED_REMOVED)
        checker.assert_not_called()

    def test_numbers_must_occur_in_evidence_before_models(self):
        checker = Mock(return_value='supported')
        result, _ = self.check(answer='Silence cures 50% of illnesses.', checker=checker)
        self.assertEqual(result.status, Outcome.UNSUPPORTED_REMOVED)
        checker.assert_not_called()

    def test_partial_answer_keeps_only_supported_original_sentences(self):
        def check(passage, claim):
            return 'unsupported' if 'cures' in claim else 'supported'
        result, calls = self.check(answer=QUOTE+' Silence cures illness.', checker=Mock(side_effect=check))
        self.assertEqual(result.status, Outcome.PARTIAL)
        self.assertNotIn('cures', result.text)
        self.assertIn(QUOTE, result.text)
        self.assertEqual(result.evidence, [QUOTE])
        self.assertEqual(len(calls), 1)
        self.assertNotIn('cures', calls[0])

    def test_no_context_is_distinct(self):
        result = ground_with_ensemble('{}', '', 'question', Mock())
        self.assertEqual(result.status, Outcome.NO_EVIDENCE)

    def test_opt_in_dispatch_preserves_existing_invocation_contract(self):
        with patch.dict('os.environ', {'GROUNDING_CHECKER': 'local_ensemble'}), patch(
                'grounding_ensemble.ground_with_ensemble', return_value='sentinel') as checker:
            self.assertEqual(ground_answer('draft', 'context', 'question', 'llm'), 'sentinel')
            checker.assert_called_once_with('draft', 'context', 'question', 'llm')

    def test_contradiction_veto_overrides_high_hhem(self):
        self.assertEqual(combine_support(.95, dict(entailment=.02, neutral=.03, contradiction=.95)), 'unsupported')

    def test_nli_neutral_does_not_veto_specialist_support(self):
        self.assertEqual(combine_support(.9, dict(entailment=.02, neutral=.97, contradiction=.01)), 'supported')

    def test_ambiguous_pair_is_uncertain(self):
        self.assertEqual(combine_support(.4, dict(entailment=.7, neutral=.29, contradiction=.01)), 'uncertain')

    def test_nonfinite_scores_raise_technical_error(self):
        with self.assertRaises(SupportClassifierError):
            combine_support(float('nan'), dict(entailment=.7, neutral=.29, contradiction=.01))



class TestReviewedHhemSource(unittest.TestCase):
    """The vendored HHEM code only runs if it matches the reviewed hashes."""
    def folder(self):
        import shutil, tempfile
        from pathlib import Path
        import grounding_ensemble
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        source = grounding_ensemble.ROOT / 'local_models/hhem-2.1-open'
        for name in grounding_ensemble.REVIEWED_HASHES:
            if not (source / name).is_file():
                self.skipTest('local HHEM files are not present')
            shutil.copy(source / name, directory / name)
        return directory

    def test_reviewed_files_are_returned_as_the_bytes_that_were_hashed(self):
        from grounding_ensemble import verified_sources
        directory = self.folder()
        reviewed = verified_sources(directory)
        self.assertEqual(reviewed['modeling_hhem_v2.py'], (directory / 'modeling_hhem_v2.py').read_bytes())

    def test_changed_code_or_config_is_refused(self):
        from grounding_ensemble import verified_sources
        for name in ['modeling_hhem_v2.py', 'config.json']:
            directory = self.folder()
            with open(directory / name, 'ab') as handle:
                handle.write(b'\nimport os  # tampered\n')
            with self.assertRaisesRegex(SupportClassifierError, 'changed'):
                verified_sources(directory)

    def test_symlinked_source_is_refused(self):
        import os
        from grounding_ensemble import verified_sources
        directory = self.folder()
        real = directory / 'real.py'
        (directory / 'modeling_hhem_v2.py').rename(real)
        try:
            os.symlink(real, directory / 'modeling_hhem_v2.py')
        except OSError:
            self.skipTest('symlinks are not available')
        with self.assertRaises(SupportClassifierError):
            verified_sources(directory)


if __name__ == '__main__':
    unittest.main()
