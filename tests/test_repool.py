import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from repool import new_candidates, grader_doc_text, merge_grades


class TestNewCandidates(unittest.TestCase):
    def test_union_first_seen_order(self):
        pools = [["a", "b"], ["b", "c"], ["c", "d"]]
        self.assertEqual(new_candidates(pools, set()), ["a", "b", "c", "d"])

    def test_excludes_already_judged(self):
        pools = [["a", "b"], ["c"]]
        self.assertEqual(new_candidates(pools, {"b"}), ["a", "c"])

    def test_dedups_across_pools(self):
        pools = [["a", "a"], ["a"]]
        self.assertEqual(new_candidates(pools, set()), ["a"])

    def test_empty(self):
        self.assertEqual(new_candidates([[], []], set()), [])


class TestGraderDocText(unittest.TestCase):
    def test_format(self):
        self.assertEqual(grader_doc_text("http://x.com", "Title body"), "URL: http://x.com\nTitle body")

    def test_empty(self):
        self.assertEqual(grader_doc_text("", ""), "URL:")


class TestMergeGrades(unittest.TestCase):
    def test_appends_new_tagged(self):
        existing = [{"qid": "q1", "doc_id": "a", "grade": 2, "source": "old"}]
        grades = [{"qid": "q1", "doc_id": "b", "grade": 1}]
        out = merge_grades(existing, grades, "new-src")
        self.assertEqual(out, [{"qid": "q1", "doc_id": "b", "grade": 1, "source": "new-src"}])

    def test_skips_already_judged(self):
        existing = [{"qid": "q1", "doc_id": "a", "grade": 2, "source": "old"}]
        grades = [{"qid": "q1", "doc_id": "a", "grade": 0}]
        self.assertEqual(merge_grades(existing, grades, "new-src"), [])

    def test_dedups_within_grades(self):
        grades = [{"qid": "q1", "doc_id": "a", "grade": 1}, {"qid": "q1", "doc_id": "a", "grade": 2}]
        out = merge_grades([], grades, "s")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["grade"], 1)


if __name__ == "__main__":
    unittest.main()
