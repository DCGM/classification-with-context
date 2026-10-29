"""No-network checks of the paid CLI wrapper's stop conditions."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from metakat_checks import run_luna_chapters as runner

sys.path.insert(0, str(runner.SOURCE))
from metakat.common.vlm import client as vlm


class FailureGuardTests(unittest.TestCase):
    def exercise(self, deterministic):
        original_init = vlm.VLMClient.__init__
        original_request = vlm.VLMClient.request_json
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'STATE', Path(directory)), patch.dict(os.environ, LUNA_BATCH='mock'):
                runner.install_guard()
                config = vlm.VLMConfig(api_url='https://example.invalid', model='test',
                                       api_key='dummy', max_attempts=1)
                client = vlm.VLMClient(config, client=object())
                calls = []

                def send(**kwargs):
                    calls.append(kwargs)
                    if deterministic:
                        return SimpleNamespace(choices=[SimpleNamespace(
                            finish_reason='length', message=SimpleNamespace(refusal=None))])
                    raise TimeoutError('mock timeout')

                client._completions_request = lambda *args: ({}, send, lambda response: '{}')
                for _ in range(3):
                    with self.assertRaises(RuntimeError):
                        client.request_json([])
                expected = 1 if deterministic else 2
                self.assertEqual(len(calls), expected)
                with runner.ledger() as state:
                    self.assertTrue(state['stopped'])
                    self.assertEqual(state['failures'], expected)
        finally:
            vlm.VLMClient.__init__ = original_init
            vlm.VLMClient.request_json = original_request

    def test_transient_failures_stop_after_two_requests(self):
        self.exercise(False)

    def test_output_limit_stops_after_first_request(self):
        self.exercise(True)

    def test_sdk_retries_are_disabled(self):
        original_init = vlm.VLMClient.__init__
        original_request = vlm.VLMClient.request_json
        try:
            with patch('openai.OpenAI') as sdk:
                runner.install_guard()
                config = vlm.VLMConfig(api_url='https://example.invalid', model='test',
                                       api_key='dummy', max_attempts=1)
                vlm.VLMClient(config)
                self.assertEqual(sdk.call_args.kwargs['max_retries'], 0)
        finally:
            vlm.VLMClient.__init__ = original_init
            vlm.VLMClient.request_json = original_request


if __name__ == '__main__':
    unittest.main()
