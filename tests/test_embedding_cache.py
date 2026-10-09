import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from retrieval import DenseIndex


class FakeDense(DenseIndex):
    calls = 0

    def encode(self, texts, is_query, show_progress=False):
        self.calls += 1
        return np.full((len(texts), 3), self.calls, dtype=np.float32)


class EmbeddingCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model.safetensors").write_bytes(b"weights-one")
        (self.model / "config.json").write_text('{"length":512}')
        self.dense = FakeDense(str(self.model), cache_dir=str(self.root / "cache"))

    def path(self, texts=None):
        return self.dense._cache_path(texts or ["a"], "corpus")

    def test_tail_and_document_boundaries_invalidate(self):
        self.assertNotEqual(self.path(["a" * 400 + "x"]), self.path(["a" * 400 + "y"]))
        self.assertNotEqual(self.path(["ab", "c"]), self.path(["a", "bc"]))
        self.assertNotEqual(self.path(["a", "b"]), self.path(["b", "a"]))

    def test_same_size_same_mtime_weights_invalidate(self):
        before = self.path()
        path = self.model / "model.safetensors"
        stat = path.stat()
        path.write_bytes(b"weights-two")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(before, self.path())

    def test_config_prefix_and_length_invalidate(self):
        before = self.path()
        (self.model / "config.json").write_text('{"length":256}')
        self.assertNotEqual(before, self.path())
        for field, value in (("passage_prefix", "passage: "), ("query_prefix", "query: "),
                             ("max_seq_length", 128)):
            before = self.path()
            setattr(self.dense, field, value)
            self.assertNotEqual(before, self.path())

    def test_aliases_of_same_local_model_share_cache(self):
        other = FakeDense(str(self.model / ".." / "model"), cache_dir=str(self.root / "cache"))
        self.assertEqual(self.path(), other._cache_path(["a"], "corpus"))

    def test_valid_cache_reused_and_corrupt_cache_rebuilt(self):
        texts = ["a", "b"]
        self.dense.build(texts)
        self.dense.build(texts)
        self.assertEqual(self.dense.calls, 1)
        Path(self.path(texts)).write_bytes(b"broken")
        self.dense.build(texts)
        self.assertEqual(self.dense.calls, 2)

    def test_legacy_or_missing_manifest_never_reused(self):
        path = self.path()
        np.save(path, np.zeros((1, 3), dtype=np.float32))
        self.dense.build(["a"])
        self.assertEqual(self.dense.calls, 1)

    def test_bad_metadata_and_invalid_matrix_rejected(self):
        self.dense.build(["a"])
        manifest_path = Path(self.path() + ".json")
        meta = json.loads(manifest_path.read_text())
        meta["identity"]["rows"] = 9
        manifest_path.write_text(json.dumps(meta))
        self.dense.build(["a"])
        self.assertEqual(self.dense.calls, 2)
        from embedding_cache import save
        with self.assertRaises(ValueError):
            save(self.path(), self.dense.cache_identity(["a"]), np.array([[np.nan]], dtype=np.float32))

    def test_disk_model_change_after_loading_fails_closed(self):
        self.dense._loaded_model_identity = self.dense._model_identity()
        (self.model / "model.safetensors").write_bytes(b"new-weights")
        with self.assertRaisesRegex(RuntimeError, "create a new DenseIndex"):
            self.dense.build(["a"])
