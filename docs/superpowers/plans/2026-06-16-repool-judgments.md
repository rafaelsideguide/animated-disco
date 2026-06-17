# Re-pooling Judgments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich `judgments.jsonl` by pooling BM25F + dense + rerank candidates and grading the new docs with Claude Code subagents (no metered API), so dense/rerank become measurable.

**Architecture:** Pure pooling/merge logic in `src/repool.py`; `scripts/build_repool_candidates.py` emits per-query new candidates; the controller grades them via subagents into `data/repool_grades.jsonl`; `scripts/merge_repool_grades.py` appends them to `judgments.jsonl`. Then re-eval.

**Tech Stack:** Python 3.11+, existing retrievers (`search`/`dense`/`rerank`), pytest, `uv`. No new dependency. Grading via Claude Code subagents (session-billed).

---

## File Structure

- **Create:** `src/repool.py` — `new_candidates`, `grader_doc_text`, `merge_grades` (pure).
- **Create:** `scripts/build_repool_candidates.py` → `data/repool_candidates.json`.
- **Create:** `scripts/merge_repool_grades.py` → appends to `data/judgments.jsonl`.
- **Create:** `tests/test_repool.py`.
- **Artifacts:** `data/repool_candidates.json` (transient), `data/repool_grades.jsonl` (transient), enriched `data/judgments.jsonl` (committed).

Constants: `POOL_DEPTH = 10` (in build script), `SOURCE = "claude-code-pooled-2026"` (in merge script).

---

## Task 1: repool.py — pooling and merge logic

**Files:** Create `src/repool.py`, `tests/test_repool.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_repool.py`:

```python
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
        self.assertEqual(out[0]["grade"], 1)  # first wins


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run python -m pytest tests/test_repool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'repool'`.

- [ ] **Step 3: Create `src/repool.py`**

```python
def new_candidates(pools: list[list[str]], judged: set) -> list[str]:
    """Union of ranked pools (first-seen order preserved), dropping any doc_id
    already in `judged` (a per-query set of already-judged doc_ids)."""
    seen = set()
    out = []
    for pool in pools:
        for doc_id in pool:
            if doc_id in judged or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
    return out


def grader_doc_text(url: str, doc_text: str) -> str:
    """Grader-facing text for a candidate: URL plus the cleaned doc text."""
    return f"URL: {url}\n{doc_text}".strip()


def merge_grades(existing_rows: list[dict], grade_rows: list[dict], source: str) -> list[dict]:
    """Return judgment rows to append: each (qid, doc_id) grade not already
    present in existing_rows (and not duplicated within grade_rows), tagged with
    source. First grade for a (qid, doc_id) wins."""
    judged = {(r["qid"], r["doc_id"]) for r in existing_rows}
    seen = set()
    out = []
    for g in grade_rows:
        key = (g["qid"], g["doc_id"])
        if key in judged or key in seen:
            continue
        seen.add(key)
        out.append({"qid": g["qid"], "doc_id": g["doc_id"], "grade": g["grade"], "source": source})
    return out
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run python -m pytest tests/test_repool.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/repool.py tests/test_repool.py
git commit -m "Add repool pooling + merge logic"
```

---

## Task 2: candidate + merge scripts

**Files:** Create `scripts/build_repool_candidates.py`, `scripts/merge_repool_grades.py`

- [ ] **Step 1: Create `scripts/build_repool_candidates.py`**

```python
import sys, os
# Strip scripts/ from sys.path so scripts/inspect.py doesn't shadow stdlib inspect.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
from pathlib import Path
from collections import defaultdict

from search import search
from dense import load_dense, dense_search
from rerank import load_doc_text, Reranker, rerank
from repool import new_candidates, grader_doc_text

DATA = Path(__file__).parent.parent / "data"
CANDIDATES_PATH = DATA / "repool_candidates.json"
POOL_DEPTH = 10


def load_queries():
    rows = {}
    with open(DATA / "queries.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def load_judged():
    judged = defaultdict(set)
    with open(DATA / "judgments.jsonl") as f:
        for line in f:
            r = json.loads(line)
            judged[r["qid"]].add(r["doc_id"])
    return judged


def main():
    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)
    vindex, embedder = load_dense(DATA)
    doc_text = load_doc_text(DATA)
    reranker = Reranker()
    reverse = {ext: i for i, ext in enumerate(index.doc_ids)}

    queries = load_queries()
    judged = load_judged()

    out = []
    total_new = 0
    for qid, row in queries.items():
        q = row["query"]
        bm = [d for d, _ in search(index, q, k=100)]
        dn = [d for d, _ in dense_search(vindex, embedder, q, k=POOL_DEPTH)]
        rr = [d for d, _ in rerank(reranker, q, bm, doc_text, k=POOL_DEPTH)]
        new = new_candidates([bm[:POOL_DEPTH], dn, rr], judged[qid])
        if not new:
            continue
        docs = []
        for doc_id in new:
            i = reverse.get(doc_id)
            url = index.doc_meta[i].get("url", "") if i is not None else ""
            docs.append({"doc_id": doc_id, "text": grader_doc_text(url, doc_text.get(doc_id, ""))})
        out.append({"qid": qid, "query": q, "docs": docs})
        total_new += len(docs)

    with open(CANDIDATES_PATH, "w") as f:
        json.dump(out, f)
    print(f"Done. {len(out)} queries, {total_new:,} new candidate docs -> {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/merge_repool_grades.py`**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path

from repool import merge_grades

DATA = Path(__file__).parent.parent / "data"
JUDGMENTS_PATH = DATA / "judgments.jsonl"
GRADES_PATH = DATA / "repool_grades.jsonl"
SOURCE = "claude-code-pooled-2026"


def main():
    existing = [json.loads(l) for l in open(JUDGMENTS_PATH)]
    grades = [json.loads(l) for l in open(GRADES_PATH)]
    new_rows = merge_grades(existing, grades, SOURCE)
    with open(JUDGMENTS_PATH, "a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Appended {len(new_rows):,} new judgments (source={SOURCE}); "
          f"judgments.jsonl now {len(existing) + len(new_rows):,} rows.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-check both scripts import**

Run: `uv run python -c "import sys, os; sys.path.insert(0, 'src'); from repool import new_candidates, grader_doc_text, merge_grades; print('ok')"`
Expected: prints `ok`. (Full candidate build runs in Task 3.)

- [ ] **Step 4: Commit**

```bash
git add scripts/build_repool_candidates.py scripts/merge_repool_grades.py
git commit -m "Add repool candidate-build and grade-merge scripts"
```

---

## Task 3: Generate candidates, grade via subagents, merge, re-eval

This task is **controller-orchestrated** (the grading is done by Claude Code
subagents, not a unit test). Execute it inline.

- [ ] **Step 1: Build candidates**

Run: `uv run python scripts/build_repool_candidates.py`
Expected: `Done. N queries, M new candidate docs -> .../data/repool_candidates.json`
(loads dense + reranker models; a few minutes). Note N and M.

- [ ] **Step 2: Grade candidates with subagents → `data/repool_grades.jsonl`**

The controller reads `data/repool_candidates.json` and dispatches grading
subagents over batches of queries (~15-20 queries per subagent; model: haiku).

Each grading subagent is given, for its batch, the query text and the list of
`{doc_id, text}` candidates, and must return **strict JSON**: an array of
`{"qid": "...", "doc_id": "...", "grade": 0|1|2}` covering **every** (qid, doc_id)
it was given. Grading rubric (matches the original): **2 = highly relevant,
1 = somewhat relevant, 0 = not relevant** to the query.

The controller parses each subagent's JSON and writes all rows to
`data/repool_grades.jsonl` (one JSON object per line). If any candidate
(qid, doc_id) is missing a grade, re-dispatch that batch. Verify at the end:
the number of grade rows equals M from Step 1.

- [ ] **Step 3: Merge grades into judgments**

Run: `uv run python scripts/merge_repool_grades.py`
Expected: `Appended K new judgments (source=claude-code-pooled-2026); judgments.jsonl now <N> rows.` (K should equal M, since all pooled candidates are new by construction.)

- [ ] **Step 4: Re-evaluate all retrievers on the enriched qrels**

Run each and record the breakdown:
```bash
uv run python scripts/run_eval.py --retriever bm25 --breakdown
uv run python scripts/run_eval.py --retriever dense --breakdown
uv run python scripts/run_eval.py --retriever bm25_rerank --breakdown
```
Expected outcome (per spec): judgment **coverage rises** for all three; **rerank
and dense improve markedly** vs their old under-measured numbers; rerank should
now beat or match BM25F. Report the before/after table.

- [ ] **Step 5: Commit the enriched judgments**

```bash
git add data/judgments.jsonl
git commit -m "Re-pool judgments via Claude Code grading (BM25F+dense+rerank top-10 union)"
```

---

## Task 4: Document re-pooling

**Files:** Modify `README.md`

- [ ] **Step 1: Add a re-pooling note to the README Notes section**

Under `## Notes`, after the existing judgment note, add:

```markdown
### Re-pooled judgments

`judgments.jsonl` was enriched (source `claude-code-pooled-2026`) by pooling the
top-10 of BM25F + dense + cross-encoder-rerank per query and grading the new
(previously unjudged) docs. Pipeline: `scripts/build_repool_candidates.py` →
grade candidates → `data/repool_grades.jsonl` → `scripts/merge_repool_grades.py`.
This de-biases the original single-system pool so semantic/rerank retrievers are
measurable. Grading was performed by Claude Code subagents rather than the
metered API.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Document re-pooled judgments in README"
```

---

## Self-Review Notes

- **Spec coverage:** `new_candidates`/`grader_doc_text`/`merge_grades` (Task 1); `build_repool_candidates.py` (BM25F+dense+rerank top-10 union, cleaned grader text) (Task 2); subagent grading → grades.jsonl (Task 3 Step 2); `merge_repool_grades.py` idempotent append with source tag (Task 2 + Task 3 Step 3); re-eval all three (Task 3 Step 4); docs (Task 4). All covered.
- **Placeholder scan:** none — code/commands concrete; the grading step is orchestration with an explicit JSON contract and rubric.
- **Type consistency:** `new_candidates(pools, judged) -> list[str]`; `grader_doc_text(url, doc_text) -> str`; `merge_grades(existing_rows, grade_rows, source) -> list[dict]`; grade rows are `{qid, doc_id, grade}` everywhere; `SOURCE`/`POOL_DEPTH` defined once. Candidate JSON shape `{qid, query, docs:[{doc_id, text}]}` consistent between build script (writer) and Task 3 grading (reader).
```
