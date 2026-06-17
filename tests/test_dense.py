import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import tempfile
import numpy as np
from dense import VectorIndex, embed_text


class TestVectorIndex(unittest.TestCase):
    def _vecs(self):
        # doc "a" on axis0, "b" on axis1, "c" near "a".
        v = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
        return v, ["a", "b", "c"]

    def test_nearest_neighbor(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        res = idx.query(np.array([1, 0, 0, 0], dtype=np.float32), k=2)
        self.assertEqual([d for d, _ in res], ["a", "c"])

    def test_similarity_descending_and_unit_for_exact(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        res = idx.query(np.array([1, 0, 0, 0], dtype=np.float32), k=3)
        sims = [s for _, s in res]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertAlmostEqual(sims[0], 1.0, places=4)

    def test_k_truncates_and_caps(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        q = np.array([1, 0, 0, 0], dtype=np.float32)
        self.assertEqual(len(idx.query(q, k=2)), 2)
        self.assertEqual(len(idx.query(q, k=99)), 3)  # capped to corpus size

    def test_save_load_roundtrip(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        with tempfile.TemporaryDirectory() as d:
            idx.save(d)
            loaded = VectorIndex.load(d)
        res = loaded.query(np.array([0, 1, 0, 0], dtype=np.float32), k=1)
        self.assertEqual(res[0][0], "b")
        self.assertEqual(loaded.doc_ids, ids)

    def test_empty_index_returns_empty(self):
        idx = VectorIndex(dim=4)
        self.assertEqual(idx.query(np.array([1, 0, 0, 0], dtype=np.float32), k=5), [])


class TestEmbedText(unittest.TestCase):
    def test_concatenates_fields_and_caps_body(self):
        out = embed_text({"title": "T", "headings": "H", "body": "B" * 1000})
        self.assertTrue(out.startswith("T H "))
        self.assertLessEqual(len(out), 512 + 8)  # body capped at 512 chars

    def test_missing_fields(self):
        self.assertEqual(embed_text({}), "")


if __name__ == "__main__":
    unittest.main()
