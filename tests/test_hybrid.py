import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from hybrid import rrf_fuse, dedup_union


class TestRrfFuse(unittest.TestCase):
    def test_single_list_scores_are_reciprocal_ranks(self):
        fused = rrf_fuse([[("a", 9.0), ("b", 5.0)]], k=60)
        self.assertEqual([d for d, _ in fused], ["a", "b"])
        self.assertAlmostEqual(fused[0][1], 1 / 60)
        self.assertAlmostEqual(fused[1][1], 1 / 61)

    def test_scores_sum_across_lists_ignoring_input_scores(self):
        # "a" is rank 0 in list 1 and rank 1 in list 2; input scores are ignored.
        fused = dict(rrf_fuse([[("a", 0.01), ("b", 0.0)],
                               [("b", 999.0), ("a", 1.0)]], k=60))
        self.assertAlmostEqual(fused["a"], 1 / 60 + 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 61 + 1 / 60)

    def test_doc_ranked_in_both_beats_doc_ranked_top_of_one(self):
        # "shared" is #2 in both lists; "solo" is #1 in only one. RRF favors agreement.
        bm25 = [("solo", 5.0), ("x", 4.0), ("shared", 3.0)]
        dense = [("y", 0.9), ("z", 0.8), ("shared", 0.7)]
        order = [d for d, _ in rrf_fuse([bm25, dense], k=60)]
        self.assertLess(order.index("shared"), order.index("solo"))

    def test_weights_scale_contributions(self):
        fused = dict(rrf_fuse([[("a", 1.0)], [("b", 1.0)]],
                              k=60, weights=[2.0, 1.0]))
        self.assertAlmostEqual(fused["a"], 2.0 / 60)
        self.assertAlmostEqual(fused["b"], 1.0 / 60)

    def test_empty_and_weight_mismatch(self):
        self.assertEqual(rrf_fuse([], k=60), [])
        self.assertEqual(rrf_fuse([[], []], k=60), [])
        with self.assertRaises(ValueError):
            rrf_fuse([[("a", 1.0)], [("b", 1.0)]], weights=[1.0])


class TestDedupUnion(unittest.TestCase):
    def test_first_seen_order_preserved_and_deduped(self):
        union = dedup_union([[("a", 1.0), ("b", 1.0)],
                             [("b", 1.0), ("c", 1.0)]])
        self.assertEqual(union, ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(dedup_union([]), [])
        self.assertEqual(dedup_union([[], []]), [])


if __name__ == "__main__":
    unittest.main()
