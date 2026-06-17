import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from rerank import rerank


class StubReranker:
    """Scores each text via a {text: score} map (missing text -> 0.0)."""
    def __init__(self, scores):
        self.scores = scores

    def score_pairs(self, query, texts):
        return [self.scores.get(t, 0.0) for t in texts]


class TestRerank(unittest.TestCase):
    def test_reorders_by_score(self):
        doc_text = {"a": "ta", "b": "tb", "c": "tc"}
        stub = StubReranker({"ta": 1.0, "tb": 3.0, "tc": 2.0})
        out = rerank(stub, "q", ["a", "b", "c"], doc_text, k=3)
        self.assertEqual([d for d, _ in out], ["b", "c", "a"])

    def test_top_k_truncates(self):
        doc_text = {"a": "ta", "b": "tb", "c": "tc"}
        stub = StubReranker({"ta": 1.0, "tb": 3.0, "tc": 2.0})
        out = rerank(stub, "q", ["a", "b", "c"], doc_text, k=2)
        self.assertEqual([d for d, _ in out], ["b", "c"])

    def test_missing_doc_scored_empty_not_dropped(self):
        doc_text = {"a": "ta"}
        stub = StubReranker({"ta": 1.0, "": 0.5})
        out = rerank(stub, "q", ["a", "z"], doc_text, k=2)
        self.assertEqual([d for d, _ in out], ["a", "z"])
        self.assertEqual(len(out), 2)

    def test_empty_candidates(self):
        self.assertEqual(rerank(StubReranker({}), "q", [], {}, k=5), [])


class TestRerankerIntegration(unittest.TestCase):
    def test_score_pairs_returns_floats_relevant_higher(self):
        try:
            from rerank import Reranker
            r = Reranker()
            scores = r.score_pairs("what is a cat", ["a cat is a small feline", "the stock market fell"])
        except (ImportError, OSError) as e:
            raise unittest.SkipTest(f"cross-encoder unavailable: {e}")
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(s, float) for s in scores))
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
