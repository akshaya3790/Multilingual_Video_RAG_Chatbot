import unittest
from types import SimpleNamespace

from inference_errors import generation_error_message


class TestInferenceErrors(unittest.TestCase):
    def test_provider_failures_have_specific_recovery_steps(self):
        for status, expected in [(401, 'Replace HF_TOKEN'), (403, 'permission'),
                                 (402, 'credits'), (429, 'Wait'),
                                 (404, 'HF_LLM_MODEL'), (503, 'temporarily')]:
            with self.subTest(status=status):
                exc = RuntimeError('secret-request-data')
                exc.response = SimpleNamespace(status_code=status)
                message = generation_error_message(exc)
                self.assertIn(expected, message)
                self.assertNotIn('secret-request-data', message)

    def test_network_failure(self):
        self.assertIn('internet connection', generation_error_message(ConnectionError()))

    def test_unexpected_failure_does_not_expose_exception(self):
        message = generation_error_message(ValueError('hf_private_token'))
        self.assertIn('server log', message)
        self.assertNotIn('hf_private_token', message)


if __name__ == '__main__':
    unittest.main()
