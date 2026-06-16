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

    def test_url_field_contributes(self):
        # A term present only in the url field is still retrievable and scored.
        idx = InvertedIndex()
        idx.add_document("u", {"url": ["revwatches"]}, {"url": "", "title": ""})
        idx.add_document("o", {"body": ["other"]}, {"url": "", "title": ""})
        idx.finalize()
        ranked = rank(["revwatches"], idx, k=5)
        self.assertEqual([d for d, _ in ranked], ["u"])

    def test_longer_body_scores_lower_for_same_tf(self):
        # Per-field length normalization (B[body]=0.75): same tf, longer body -> lower score.
        idx = InvertedIndex()
        idx.add_document("short", {"body": ["python"]}, {"url": "", "title": ""})
        idx.add_document("long", {"body": ["python"] + ["filler"] * 20}, {"url": "", "title": ""})
        idx.finalize()
        ranked = rank(["python"], idx, k=2)
        self.assertEqual(ranked[0][0], "short")


if __name__ == "__main__":
    unittest.main()
