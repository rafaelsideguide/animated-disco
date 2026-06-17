import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from route import bm25_margin, escalation_target, CostRouter


def _confident():
    # s0=10, s_10=1.0 -> margin 0.9
    return [("d0", 10.0)] + [(f"d{i}", 1.0) for i in range(1, 11)]


def _flat():
    # 11 equal scores -> margin 0.0
    return [(f"d{i}", 10.0) for i in range(11)]


class RerankSpy:
    def __init__(self):
        self.calls = 0

    def __call__(self, query, candidate_ids):
        self.calls += 1
        return [(c, 1.0) for c in candidate_ids]


class TestBm25Margin(unittest.TestCase):
    def test_margin_basic(self):
        self.assertAlmostEqual(bm25_margin(_confident(), k=10), 0.9)

    def test_flat_is_zero(self):
        self.assertEqual(bm25_margin(_flat(), k=10), 0.0)

    def test_empty_is_zero(self):
        self.assertEqual(bm25_margin([], k=10), 0.0)

    def test_single_is_zero(self):
        self.assertEqual(bm25_margin([("a", 5.0)], k=10), 0.0)

    def test_nonpositive_top_is_zero(self):
        self.assertEqual(bm25_margin([("a", 0.0), ("b", 0.0)], k=1), 0.0)


class TestEscalationTarget(unittest.TestCase):
    def test_ascii_to_rerank(self):
        self.assertEqual(escalation_target("docker engine api"), "bm25_rerank")

    def test_nonascii_to_rrf(self):
        self.assertEqual(escalation_target("actualités du monde"), "hybrid_rrf")


class TestCostRouter(unittest.TestCase):
    def _router(self, bm25_list, dense_list, spy, tau=0.3):
        return CostRouter(
            bm25_fn=lambda q: bm25_list,
            dense_fn=lambda q: dense_list,
            rerank_fn=spy,
            tau=tau,
        )

    def test_confident_returns_bm25_no_rerank(self):
        spy = RerankSpy()
        r = self._router(_confident(), [], spy)
        out = r.retrieve("anything")
        self.assertEqual([d for d, _ in out], [d for d, _ in _confident()])
        self.assertEqual(spy.calls, 0)
        self.assertEqual(r.stats["bm25_only"], 1)
        self.assertEqual(r.stats["cross_encoder_calls"], 0)

    def test_uncertain_ascii_escalates_to_rerank(self):
        spy = RerankSpy()
        r = self._router(_flat(), [], spy)
        r.retrieve("plain ascii query")
        self.assertEqual(spy.calls, 1)
        self.assertEqual(r.stats["bm25_rerank"], 1)
        self.assertEqual(r.stats["cross_encoder_calls"], 1)
        self.assertEqual(r.stats["pairs_scored"], 11)

    def test_uncertain_nonascii_escalates_to_rrf_no_rerank(self):
        spy = RerankSpy()
        r = self._router(_flat(), [("x", 0.9)], spy)
        r.retrieve("notícias económicas")
        self.assertEqual(spy.calls, 0)
        self.assertEqual(r.stats["hybrid_rrf"], 1)
        self.assertEqual(r.stats["cross_encoder_calls"], 0)


if __name__ == "__main__":
    unittest.main()
