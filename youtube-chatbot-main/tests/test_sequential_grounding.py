"""Model lifecycle tests with small fakes; no weights, torch, or inference loaded."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
import weakref

from factual_support import SupportClassifierError
from grounding_ensemble import LocalSupportEnsemble


class Ledger:
    def __init__(self):
        self.live = weakref.WeakSet()
        self.events = []
        self.maximum = 0

    def weight(self, name):
        if self.live:
            raise AssertionError('Models overlap')
        ledger = self
        class Weight:
            def __del__(self):
                ledger.events.append(name + '_released')
        value = Weight()
        self.live.add(value)
        self.maximum = max(self.maximum, len(self.live))
        self.events.append(name + '_loaded')
        return value


def fixture(ledger, *, hhem_failure=None, nli_failure=None, long_input=False, held=None):
    checker = LocalSupportEnsemble()

    class FakeNLI:
        _model = None
        _tokenizer = None

        def _load(self):
            if checker._hhem is not None:
                raise AssertionError('HHEM remains loaded')
            self._model = ledger.weight('nli')
            self._tokenizer = object()

        def classify(self, passage, claim):
            ledger.events.append('nli_score_' + claim)
            if nli_failure:
                raise nli_failure
            return SimpleNamespace(label='entailment', scores=dict(entailment=.99, contradiction=.005, neutral=.005))

    checker._nli = FakeNLI()

    def load_hhem():
        if checker._nli._model is not None:
            raise AssertionError('NLI remains loaded')
        weight = ledger.weight('hhem')
        weight.prompt = '{text1} {text2}'
        weight.tokenzier = lambda *args, **kwargs: {'input_ids': [1] * (513 if long_input else 5)}
        def predict(pairs):
            claim = pairs[0][1]
            ledger.events.append('hhem_score_' + claim)
            if hhem_failure:
                raise hhem_failure
            time.sleep(.002)
            return [.9]
        weight.predict = predict
        checker._hhem = weight
        if held is not None:
            held.append(weight)

    checker._load_hhem = load_hhem
    return checker


class TestSequentialGrounding(unittest.TestCase):
    def test_all_hhem_scores_precede_release_and_nli_load(self):
        ledger = Ledger()
        checker = fixture(ledger)
        with patch('grounding_ensemble.available_memory_mib', return_value=3000):
            self.assertEqual(checker.check_many([('source', 'one'), ('source', 'two')]), ['supported', 'supported'])
        self.assertLess(ledger.events.index('hhem_score_two'), ledger.events.index('hhem_released'))
        self.assertLess(ledger.events.index('hhem_released'), ledger.events.index('nli_loaded'))
        self.assertEqual(ledger.maximum, 1)
        self.assertEqual(len(ledger.live), 0)
        self.assertIsNone(checker._hhem)
        self.assertIsNone(checker._nli._model)
        self.assertIsNone(checker._nli._tokenizer)

    def test_single_pair_contract_is_preserved(self):
        checker = fixture(Ledger())
        with patch('grounding_ensemble.available_memory_mib', return_value=3000):
            self.assertEqual(checker.check('source', 'claim'), 'supported')

    def test_hhem_failure_releases_model_and_never_loads_nli(self):
        ledger = Ledger()
        checker = fixture(ledger, hhem_failure=RuntimeError('private marker'))
        with patch('grounding_ensemble.available_memory_mib', return_value=3000):
            with self.assertRaises(SupportClassifierError) as failure:
                checker.check('source', 'claim')
        self.assertNotIn('private marker', str(failure.exception))
        self.assertNotIn('nli_loaded', ledger.events)
        self.assertIsNone(checker._hhem)
        self.assertIsNone(checker._nli._model)

    def test_nli_timeout_releases_both_models(self):
        ledger = Ledger()
        checker = fixture(ledger, nli_failure=TimeoutError('private marker'))
        with patch('grounding_ensemble.available_memory_mib', return_value=3000):
            with self.assertRaises(SupportClassifierError):
                checker.check('source', 'claim')
        self.assertIsNone(checker._hhem)
        self.assertIsNone(checker._nli._model)
        self.assertIn('nli_released', ledger.events)

    def test_external_model_reference_blocks_next_load(self):
        ledger, held = Ledger(), []
        checker = fixture(ledger, held=held)
        with patch('grounding_ensemble.available_memory_mib', return_value=3000):
            with self.assertRaises(SupportClassifierError):
                checker.check('source', 'claim')
        self.assertNotIn('nli_loaded', ledger.events)
        held.clear()

    def test_second_stage_keeps_1500_mib_guard(self):
        ledger = Ledger()
        checker = fixture(ledger)
        def memory():
            return 1499 if 'hhem_released' in ledger.events else 3000
        with patch('grounding_ensemble.available_memory_mib', side_effect=memory):
            with self.assertRaises(SupportClassifierError):
                checker.check('source', 'claim')
        self.assertNotIn('nli_loaded', ledger.events)

    def test_initial_1500_mib_guard_runs_before_real_loader(self):
        checker = LocalSupportEnsemble()
        with patch('grounding_ensemble.available_memory_mib', return_value=1499):
            with self.assertRaises(SupportClassifierError):
                checker.check('source', 'claim')
        self.assertIsNone(checker._hhem)
        self.assertIsNone(checker._nli._model)

    def test_256_mib_inference_guard_remains(self):
        ledger = Ledger()
        checker = fixture(ledger)
        with patch('grounding_ensemble.available_memory_mib', return_value=255):
            with self.assertRaises(SupportClassifierError):
                checker.check('source', 'claim')
        self.assertNotIn('nli_loaded', ledger.events)

    def test_long_input_uncertain_does_not_load_second_model(self):
        ledger = Ledger()
        checker = fixture(ledger, long_input=True)
        with patch('grounding_ensemble.available_memory_mib', return_value=3000):
            self.assertEqual(checker.check('source', 'claim'), 'uncertain')
        self.assertNotIn('nli_loaded', ledger.events)
        self.assertEqual(len(ledger.live), 0)

    def test_empty_batch_does_not_load_models(self):
        checker = fixture(Ledger())
        self.assertEqual(checker.check_many([]), [])

    def test_concurrent_instances_cannot_overlap_weights(self):
        ledger = Ledger()
        first, second = fixture(ledger), fixture(ledger)
        with patch('grounding_ensemble.available_memory_mib', return_value=3000):
            with ThreadPoolExecutor(max_workers=2) as workers:
                futures = [workers.submit(checker.check, 'source', 'claim') for checker in [first, second]]
                self.assertEqual([future.result() for future in futures], ['supported', 'supported'])
        self.assertEqual(ledger.maximum, 1)
        self.assertEqual(len(ledger.live), 0)


if __name__ == '__main__':
    unittest.main()
