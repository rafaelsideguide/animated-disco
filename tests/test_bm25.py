import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from index import InvertedIndex
from bm25 import rank


class TestRankTieBreak(unittest.TestCase):
    def _build(self, docs):
        idx = InvertedIndex()
        for doc_id, tokens in docs:
            idx.add_document(doc_id, {"body": tokens}, {"url": "", "title": ""})
        idx.finalize()
        return idx

    def test_equal_scores_keep_insertion_order(self):
        idx = self._build([("a", ["x"]), ("b", ["x"]), ("c", ["x"])])
        ranked = rank(["x"], idx, k=3)
        self.assertEqual([doc for doc, _ in ranked], ["a", "b", "c"])
        self.assertEqual(ranked[0][1], ranked[1][1])

    def test_higher_score_ranks_first(self):
        idx = self._build([("a", ["x", "x"]), ("b", ["x"])])
        ranked = rank(["x"], idx, k=2)
        self.assertEqual(ranked[0][0], "a")

    def test_k_truncates(self):
        idx = self._build([("a", ["x"]), ("b", ["x"]), ("c", ["x"])])
        self.assertEqual(len(rank(["x"], idx, k=2)), 2)


class TestBM25FFieldBoost(unittest.TestCase):
    def test_title_match_outranks_body_match(self):
        idx = InvertedIndex()
        idx.add_document("t", {"title": ["python"], "body": ["filler", "filler"]}, {"url": "", "title": ""})
        idx.add_document("b", {"body": ["python", "filler", "filler"]}, {"url": "", "title": ""})
        idx.finalize()
        ranked = rank(["python"], idx, k=2)
        self.assertEqual(ranked[0][0], "t")


if __name__ == "__main__":
    unittest.main()
