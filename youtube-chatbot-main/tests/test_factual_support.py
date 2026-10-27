"""Classifier contract tests; model predictions are mocked, not live verification."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from factual_support import FactualSupportClassifier, SupportClassifierError


class TestFactualSupport(unittest.TestCase):
    def test_missing_model_is_explicit_and_does_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            classifier = FactualSupportClassifier(directory)
            with self.assertRaises(SupportClassifierError):
                classifier.classify('A transcript passage.', 'A claim.')

    def test_empty_input_is_a_technical_error(self):
        for passage, claim in [('', 'claim'), ('text', ''), (None, 'claim')]:
            with self.assertRaises(SupportClassifierError):
                FactualSupportClassifier().classify(passage, claim)

    def fixture(self, length=8, logits=(0., 8., 0.)):
        import torch
        classifier = FactualSupportClassifier()
        classifier._labels = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        classifier._tokenizer = Mock(return_value={'input_ids': torch.ones((1, length), dtype=torch.long)})
        classifier._model = Mock(return_value=SimpleNamespace(logits=torch.tensor([logits])))
        classifier._model.config.max_position_embeddings = 512
        return classifier

    def test_uses_configuration_label_mapping_and_passage_first(self):
        classifier = self.fixture()
        decision = classifier.classify('passage', 'claim')
        self.assertEqual(decision.label, 'entailment')
        self.assertGreater(decision.scores['entailment'], .99)
        classifier._tokenizer.assert_called_once_with('passage', 'claim', return_tensors='pt', truncation=False)

    def test_long_input_is_uncertain_and_never_silently_truncated(self):
        classifier = self.fixture(length=513)
        self.assertEqual(classifier.classify('passage', 'claim').label, 'uncertain')
        classifier._model.assert_not_called()

    def test_inference_error_does_not_leak_exception_text(self):
        classifier = self.fixture()
        classifier._model.side_effect = RuntimeError('secret test marker')
        with self.assertRaises(SupportClassifierError) as result:
            classifier.classify('passage', 'claim')
        self.assertNotIn('secret test marker', str(result.exception))

    def test_nonfinite_output_is_a_technical_error(self):
        classifier = self.fixture(logits=(float('nan'), 1., 0.))
        with self.assertRaises(SupportClassifierError):
            classifier.classify('passage', 'claim')


if __name__ == '__main__':
    unittest.main()
