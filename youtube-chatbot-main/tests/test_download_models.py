"""Model download script checks; downloads are mocked, nothing touches the network."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_models
from factual_support import SupportClassifierError


class TestDownloadModels(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        source = Path(download_models.__file__).parent / 'local_models' / 'hhem-2.1-open'
        (self.root / 'hhem-2.1-open').mkdir()
        for name in ['configuration_hhem_v2.py', 'modeling_hhem_v2.py', 'config.json']:
            if not (source / name).is_file():
                self.skipTest('reviewed HHEM files are not present')
            shutil.copy(source / name, self.root / 'hhem-2.1-open' / name)

    def run_with(self, weights):
        def fake_download(repo, filename, revision, local_dir):
            self.assertEqual(revision, download_models.HHEM_REVISION)
            Path(local_dir, filename).write_bytes(weights)
        with patch.object(download_models, 'ROOT', self.root), \
             patch.object(download_models, 'hf_hub_download', side_effect=fake_download) as hub, \
             patch.object(download_models, 'snapshot_download') as snapshots:
            download_models.main()
        return hub, snapshots

    def test_pinned_revisions_are_requested(self):
        with patch.object(download_models, 'HHEM_WEIGHTS_SHA256', download_models.hashlib.sha256(b'w').hexdigest()):
            _, snapshots = self.run_with(b'w')
        revisions = {call.args[0]: call.kwargs['revision'] for call in snapshots.call_args_list}
        self.assertEqual(revisions['cross-encoder/nli-deberta-v3-xsmall'], download_models.NLI_REVISION)
        self.assertEqual(revisions['google/flan-t5-base'], download_models.FOUNDATION_REVISION)

    def test_weights_with_the_wrong_hash_are_refused(self):
        with self.assertRaisesRegex(SystemExit, 'do not match'):
            self.run_with(b'not the reviewed weights')

    def test_changed_hhem_code_stops_before_downloading(self):
        with open(self.root / 'hhem-2.1-open' / 'modeling_hhem_v2.py', 'a') as handle:
            handle.write('\n# changed\n')
        with self.assertRaises(SupportClassifierError):
            self.run_with(b'w')


if __name__ == '__main__':
    unittest.main()
