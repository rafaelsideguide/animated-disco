# Cross-Encoder Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerank BM25F's top-100 with a multilingual cross-encoder, pulling already-retrieved relevant docs into the top-10.

**Architecture:** `src/rerank.py` splits a lazy `Reranker` (sentence-transformers `CrossEncoder`) from a model-free `rerank()` ordering function. `scripts/build_doc_store.py` persists `doc_id → text` to `data/doc_text.pkl`. `run_eval.py` gains a `bm25_rerank` retriever that scores BM25F's top-100 query-doc pairs and reorders them. BM25F/dense code untouched.

**Tech Stack:** Python 3.11+, `sentence-transformers` `CrossEncoder` (no new dependency), pytest, `uv`.

---

## File Structure

- **Create:** `src/rerank.py` — `Reranker`, `rerank`, `load_doc_text`, constants.
- **Create:** `scripts/build_doc_store.py` — builds `data/doc_text.pkl`.
- **Create:** `tests/test_rerank.py` — `rerank()` stub tests + skip-if-unavailable `Reranker` test.
- **Modify:** `scripts/run_eval.py` — add `bm25_rerank` to `--retriever`.
- **Artifact:** `data/doc_text.pkl` (LFS).

Constants in `src/rerank.py`: `RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"`, `RERANK_BODY_CHARS = 512`.

---

## Task 1: rerank.py — reranker + ordering logic

**Files:** Create `src/rerank.py`, `tests/test_rerank.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rerank.py`:

```python
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
        # "z" has no doc_text -> empty string -> stub scores "" as 0.5, still ranked.
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run python -m pytest tests/test_rerank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rerank'`.

- [ ] **Step 3: Create `src/rerank.py`**

```python
import pickle
from pathlib import Path

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_BODY_CHARS = 512
_DOC_TEXT_FILE = "doc_text.pkl"


class Reranker:
    """Lazy sentence-transformers CrossEncoder wrapper."""

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._ensure().predict([(query, t) for t in texts])
        return [float(s) for s in scores]


def rerank(reranker, query: str, candidate_ids: list[str], doc_text: dict, k: int = 10) -> list[tuple[str, float]]:
    """Reorder candidate_ids by cross-encoder score (descending), keep top-k.
    Missing doc_ids resolve to empty text (scored, not dropped)."""
    if not candidate_ids:
        return []
    texts = [doc_text.get(cid, "") for cid in candidate_ids]
    scores = reranker.score_pairs(query, texts)
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:k]
    return [(cid, float(s)) for cid, s in ranked]


def load_doc_text(directory) -> dict:
    with open(Path(directory) / _DOC_TEXT_FILE, "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run python -m pytest tests/test_rerank.py -v`
Expected: all PASS (integration test PASSES since the model is cached, else SKIPPED).

- [ ] **Step 5: Commit**

```bash
git add src/rerank.py tests/test_rerank.py
git commit -m "Add cross-encoder Reranker and rerank() ordering"
```

---

## Task 2: doc-text store build script

**Files:** Create `scripts/build_doc_store.py`

- [ ] **Step 1: Create the script**

```python
import sys, os
# Strip scripts/ from sys.path so scripts/inspect.py doesn't shadow stdlib inspect.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
from pathlib import Path

from parse import parse_document
from rerank import RERANK_BODY_CHARS

DATA = Path(__file__).parent.parent / "data"
CORPUS_PATH = DATA / "corpus.jsonl"
DOC_TEXT_PATH = DATA / "doc_text.pkl"


def doc_store_text(doc: dict) -> str:
    fields = parse_document(doc)
    body = fields["body"][:RERANK_BODY_CHARS]
    return f"{fields['title']} {fields['headings']} {body}".strip()


def main():
    doc_text = {}
    print("Building doc-text store...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            doc = json.loads(line)
            doc_text[doc["id"]] = doc_store_text(doc)
            if i % 10_000 == 0:
                print(f"  processed {i:,} docs...")
    with open(DOC_TEXT_PATH, "wb") as f:
        pickle.dump(doc_text, f)
    print(f"Done. {len(doc_text):,} docs -> {DOC_TEXT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check imports (do not run the full build yet)**

Run: `uv run python -c "import sys, os; sys.path.insert(0, 'src'); import json, pickle; from parse import parse_document; print(parse_document({'title':'T','url':'https://x.com/a','markdown':'# H\nbody'})['title'])"`
Expected: prints `T` (confirms the parse pipeline the script depends on works). The full build runs in Task 4.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_doc_store.py
git commit -m "Add doc-text store build script for reranking"
```

---

## Task 3: Wire bm25_rerank into run_eval

**Files:** Modify `scripts/run_eval.py`

- [ ] **Step 1: Add the `bm25_rerank` choice and branch**

In `scripts/run_eval.py`, replace this block in `main()`:
```python
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    args = parser.parse_args()

    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    if args.retriever == "dense":
        from dense import load_dense, dense_search
        vindex, embedder = load_dense(DATA)
        def retrieve(q):
            return dense_search(vindex, embedder, q, k=100)
    else:
        import pickle
        from search import search
        with open(DATA / "index.pkl", "rb") as f:
            index = pickle.load(f)
        def retrieve(q):
            return search(index, q, k=100)
```
with:
```python
    parser.add_argument("--retriever", choices=["bm25", "dense", "bm25_rerank"], default="bm25")
    args = parser.parse_args()

    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    if args.retriever == "dense":
        from dense import load_dense, dense_search
        vindex, embedder = load_dense(DATA)
        def retrieve(q):
            return dense_search(vindex, embedder, q, k=100)
    elif args.retriever == "bm25_rerank":
        import pickle
        from search import search
        from rerank import load_doc_text, Reranker, rerank
        with open(DATA / "index.pkl", "rb") as f:
            index = pickle.load(f)
        doc_text = load_doc_text(DATA)
        reranker = Reranker()
        def retrieve(q):
            candidates = [d for d, _ in search(index, q, k=100)]
            return rerank(reranker, q, candidates, doc_text, k=100)
    else:
        import pickle
        from search import search
        with open(DATA / "index.pkl", "rb") as f:
            index = pickle.load(f)
        def retrieve(q):
            return search(index, q, k=100)
```
Leave the rest of `main()` unchanged.

- [ ] **Step 2: Verify the default bm25 path is unchanged**

Run: `uv run python scripts/run_eval.py 2>&1 | tail -2`
Expected: `NDCG@10=0.65  MRR=0.84  Recall@100=0.83  (n=197)`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_eval.py
git commit -m "run_eval: add bm25_rerank retriever (cross-encoder over BM25F top-100)"
```

---

## Task 4: Build doc store and evaluate reranking

**Files:** Artifact `data/doc_text.pkl`

- [ ] **Step 1: Build the doc-text store**

Run: `uv run python scripts/build_doc_store.py`
Expected: streams the corpus, ends with `Done. 50,000 docs -> .../data/doc_text.pkl`.

- [ ] **Step 2: LFS-track the artifact**

Run:
```bash
printf 'data/doc_text.pkl filter=lfs diff=lfs merge=lfs -text\n' >> .gitattributes
git check-attr filter -- data/doc_text.pkl
```
Expected: `data/doc_text.pkl: filter: lfs`.

- [ ] **Step 3: Evaluate reranking**

Run: `uv run python scripts/run_eval.py --retriever bm25_rerank --breakdown`
Expected: overall + per-bucket table. Record it. (~100 cross-encoder passes/query × 197 — a few minutes.)

- [ ] **Step 4: Check success criteria vs BM25F**

BM25F reference: overall 0.65; natural 0.66; keyword 0.70; paraphrase 0.44; hyphenated 0.68; code-identifier 0.61; non-english 0.82.

- **Primary:** overall NDCG@10 > 0.65.
- **Guard:** non-english must not regress materially (the reason for the multilingual model). If overall or non-english drops, STOP and report before committing — investigate doc-text length or model choice.

- [ ] **Step 5: Commit the artifact**

```bash
git add .gitattributes data/doc_text.pkl
git commit -m "Build doc-text store; cross-encoder rerank eval"
```

---

## Self-Review Notes

- **Spec coverage:** `Reranker` lazy CrossEncoder (Task 1); `rerank()` reorder/top-k/missing-doc (Task 1); `load_doc_text` (Task 1); `build_doc_store.py` → doc_text.pkl with title+headings+body[:512] (Task 2); `run_eval --retriever bm25_rerank` BM25F top-100 → rerank top-100 (Task 3); build + eval + non-english guard (Task 4); LFS artifact (Task 4). All covered.
- **Placeholder scan:** none — all code/commands concrete.
- **Type consistency:** `Reranker.score_pairs(query, texts) -> list[float]`; `rerank(reranker, query, candidate_ids, doc_text, k) -> [(doc_id, float)]`; `load_doc_text(dir) -> dict`; `RERANK_MODEL`/`RERANK_BODY_CHARS`/`_DOC_TEXT_FILE` defined once in rerank.py and reused by build_doc_store.py. The `retrieve` closure returns `[(doc_id, score)]`, consistent with bm25/dense branches.
```
