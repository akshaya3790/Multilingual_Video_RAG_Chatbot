"""The helper scripts and memory checks work on any operating system."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import system_memory

HERE = Path(__file__).resolve().parent.parent  # the project folder


class TestSystemMemory(unittest.TestCase):
    def test_this_machine_reports_sensible_numbers(self):
        total, available = system_memory.memory_mib()
        self.assertGreater(total, 0)
        self.assertTrue(0 <= available <= total)

    def test_macos_numbers_come_from_sysctl_and_vm_stat(self):
        outputs = {
            'sysctl': '17179869184\n',
            'vm_stat': ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
                        'Pages free:   100000.\nPages active:  5.\n'
                        'Pages inactive:   50000.\nPages speculative:   10000.\n'),
        }
        run = lambda command, **kw: subprocess.CompletedProcess(command, 0, outputs[command[0]], '')
        with patch.object(system_memory.os, 'name', 'posix'), patch.object(system_memory.sys, 'platform', 'darwin'), \
             patch.object(system_memory.subprocess, 'run', side_effect=run):
            self.assertEqual(system_memory.memory_mib(), (16384, 160000 * 16384 // 2**20))

    def test_scripts_no_longer_need_windows(self):
        for name in ['inspect_nli_environment.py', 'evaluate_factual_support.py', 'grounding_ensemble.py']:
            self.assertNotIn('windll', (HERE / name).read_text(encoding='utf-8'), name)

    def test_inspection_script_runs_here(self):
        result = subprocess.run([sys.executable, str(HERE / 'inspect_nli_environment.py')],
                                capture_output=True, text=True, cwd=HERE, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertIn('RAM MiB total/available:', result.stdout)

    def test_benchmark_cases_ship_with_the_project_and_missing_files_are_explained(self):
        cases = json.loads((HERE / 'benchmarks' / 'factual_support_cases.json').read_text(encoding='utf-8'))
        self.assertTrue(cases and {'id', 'premise', 'hypothesis', 'expected_entailment'} <= set(cases[0]))
        result = subprocess.run([sys.executable, str(HERE / 'evaluate_factual_support.py'), 'no_such_file.json'],
                                capture_output=True, text=True, cwd=HERE, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Case file not found', result.stderr)


class TestLogging(unittest.TestCase):
    def test_lines_carry_level_and_request_id(self):
        import io
        import logging
        import app_logging
        logger = app_logging.setup_logging()
        stream = io.StringIO()
        handler = logger.handlers[0]
        old = handler.setStream(stream)
        try:
            request_id = app_logging.new_request_id()
            logger.getChild('test').warning('something happened')
        finally:
            handler.setStream(old)
        line = stream.getvalue()
        self.assertIn('WARNING', line)
        self.assertIn(f'[{request_id}]', line)
        self.assertIn('ytchat.test: something happened', line)


if __name__ == '__main__':
    unittest.main()
