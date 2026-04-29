import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from index import InvertedIndex
from bm25 import rank


class TestSmokePipeline(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex()
        self.index.add_document("a", ["python", "web", "framework"], {"url": "http://example.com/a", "title": "Python Web Framework"})
        self.index.add_document("b", ["python", "data", "science"], {"url": "http://example.com/b", "title": "Python Data Science"})
        self.index.add_document("c", ["javascript", "web", "frontend"], {"url": "http://example.com/c", "title": "JavaScript Web Frontend"})
        self.index.finalize()

    def test_python_web_returns_doc_a_first(self):
        query_tokens = ["python", "web"]
        results = rank(query_tokens, self.index, k=3)
        self.assertTrue(len(results) > 0, "Expected at least one result")
        top_doc_id = results[0][0]
        self.assertEqual(top_doc_id, "a", f"Expected doc 'a' first, got '{top_doc_id}'")


if __name__ == "__main__":
    unittest.main()
