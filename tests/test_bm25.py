import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from index import InvertedIndex
from bm25 import rank


class TestRankTieBreak(unittest.TestCase):
    """rank() uses heapq.nlargest; pin that its ordering (incl. ties) matches the
    stable sorted(..., reverse=True)[:k] it replaced."""

    def _build(self, docs):
        idx = InvertedIndex()
        for doc_id, tokens in docs:
            idx.add_document(doc_id, tokens, {"url": "", "title": ""})
        idx.finalize()
        return idx

    def test_equal_scores_keep_insertion_order(self):
        # Three docs identical w.r.t. the query term => identical scores. The tie
        # must resolve to first-indexed-first, matching stable sort behavior.
        idx = self._build([("a", ["x"]), ("b", ["x"]), ("c", ["x"])])
        ranked = rank(["x"], idx, k=3)
        self.assertEqual([doc for doc, _ in ranked], ["a", "b", "c"])
        self.assertEqual(ranked[0][1], ranked[1][1])  # genuinely tied

    def test_higher_score_ranks_first(self):
        # Doc "a" has the term twice => higher tf => strictly higher score.
        idx = self._build([("a", ["x", "x"]), ("b", ["x"])])
        ranked = rank(["x"], idx, k=2)
        self.assertEqual(ranked[0][0], "a")

    def test_k_truncates(self):
        idx = self._build([("a", ["x"]), ("b", ["x"]), ("c", ["x"])])
        self.assertEqual(len(rank(["x"], idx, k=2)), 2)


if __name__ == "__main__":
    unittest.main()
