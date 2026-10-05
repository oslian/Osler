"""Synthetic restoration checks; no research/patient data."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.restore_research_assets import main


class PublicationAssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.old)

    def fixture(self, name='data/candidate.pt'):
        payload = b'synthetic parameter fixture'
        with tarfile.open('weights.tar.gz', 'w:gz') as tar:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        asset = Path('weights.tar.gz')
        Path('RESEARCH_ASSETS.json').write_text(json.dumps({'assets': [{
            'name': asset.name, 'bytes': asset.stat().st_size,
            'sha256': hashlib.sha256(asset.read_bytes()).hexdigest(), 'files': [name]}]}))
        Path('RESEARCH_WEIGHTS_MANIFEST.json').write_text(json.dumps({
            'files': [{'path': name, 'bytes': len(payload),
                       'sha256': hashlib.sha256(payload).hexdigest()}], 'reader_files': []}))
        return payload

    def run_restore(self, *args):
        with patch.object(sys, 'argv', ['restore', '--asset-dir', '.', *args]):
            main()

    def test_verify_only_then_restore(self):
        payload = self.fixture()
        self.run_restore()
        self.assertFalse(Path('data/candidate.pt').exists())
        self.run_restore('--extract')
        self.assertEqual(Path('data/candidate.pt').read_bytes(), payload)
        self.run_restore('--extract')

    def test_refuses_different_existing_file(self):
        self.fixture()
        Path('data').mkdir()
        Path('data/candidate.pt').write_bytes(b'existing')
        with self.assertRaisesRegex(ValueError, 'Refusing overwrite'):
            self.run_restore('--extract')
        self.assertEqual(Path('data/candidate.pt').read_bytes(), b'existing')

    def test_does_not_delete_preexisting_temporary_file(self):
        self.fixture()
        Path('data').mkdir()
        pending = Path('data/candidate.pt.download-tmp')
        pending.write_bytes(b'other operation')
        with self.assertRaises(FileExistsError):
            self.run_restore('--extract')
        self.assertEqual(pending.read_bytes(), b'other operation')

    def test_rejects_traversal_and_bad_hash(self):
        self.fixture('../escape.pt')
        with self.assertRaisesRegex(ValueError, 'Unsafe member'):
            self.run_restore('--extract')
        self.fixture()
        Path('weights.tar.gz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.run_restore()


if __name__ == '__main__':
    unittest.main()
