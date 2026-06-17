# Hybrid Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two hybrid retrievers — RRF fusion of BM25F+dense, and a cross-encoder over the BM25F∪dense candidate union — and benchmark them against the existing `bm25`, `dense`, and `bm25_rerank`.

**Architecture:** A new `src/hybrid.py` holds rank-fusion (`rrf_fuse`) and a `dedup_union` helper. `run_eval.py` gains two `--retriever` choices that compose existing `search()`, `dense_search()`, and `rerank()` — no new model code, no index rebuild.

**Tech Stack:** Python 3.11+, pytest (unittest style), existing `index.pkl` / hnswlib dense index / `doc_text.pkl`.

## Global Constraints

- `requires-python = ">=3.11"`.
- Tests use `unittest` classes run under pytest; import `src/` modules via `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))` (match `tests/test_rerank.py`).
- Retrieve functions return `list[tuple[str, float]]` `(doc_id, score)` sorted best-first.
- RRF defaults: `k=60`, equal weights. No index rebuild — reuse `data/index.pkl`, `data/hnsw.bin`+`data/dense_meta.pkl`, `data/doc_text.pkl`.
- No new dependencies.

---

### Task 1: `src/hybrid.py` — rank fusion + union helper

**Files:**
- Create: `src/hybrid.py`
- Test: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: nothing (pure functions over ranked lists).
- Produces:
  - `rrf_fuse(ranked_lists: list[list[tuple[str, float]]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]` — fused `(doc_id, rrf_score)` sorted descending.
  - `dedup_union(ranked_lists: list[list[tuple[str, float]]]) -> list[str]` — first-seen-order unique doc_ids across all lists.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hybrid.py`:

```python
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
        order = [d for d, _ in rrf_fuse([bm25, dense], k=1)]
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hybrid'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/hybrid.py`:

```python
# Rank-fusion utilities for combining ranked retrieval lists.


def rrf_fuse(ranked_lists, k=60, weights=None):
    """Reciprocal-rank fusion over already-sorted ranked lists.

    Score(d) = sum_i w_i / (k + rank_i(d)), rank_i 0-based within list i. A doc
    absent from a list contributes nothing from it. Input scores are IGNORED —
    only ranks matter, which makes RRF robust to incomparable score scales
    (BM25F magnitudes vs. cosine similarity). Returns (doc_id, score) sorted
    descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
        )

    scores = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def dedup_union(ranked_lists):
    """First-seen-order unique doc_ids across all ranked lists."""
    seen = set()
    union = []
    for ranked in ranked_lists:
        for doc_id, _score in ranked:
            if doc_id not in seen:
                seen.add(doc_id)
                union.append(doc_id)
    return union
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hybrid.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/hybrid.py tests/test_hybrid.py
git commit -m "Add hybrid rank-fusion: rrf_fuse + dedup_union"
```

---

### Task 2: Wire `hybrid_rrf` and `hybrid_rerank` into `run_eval.py`

**Files:**
- Modify: `scripts/run_eval.py` (the `--retriever` arg `choices` and the retriever-selection `if/elif` chain in `main()`)

**Interfaces:**
- Consumes: `rrf_fuse`, `dedup_union` from `hybrid`; `search` from `search`; `load_dense`, `dense_search` from `dense`; `load_doc_text`, `Reranker`, `rerank` from `rerank`.
- Produces: two new CLI retrievers, each `retrieve(q) -> list[tuple[str, float]]`.

- [ ] **Step 1: Add the new choices to the argparse argument**

In `scripts/run_eval.py`, change the `--retriever` line:

```python
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "bm25_rerank", "hybrid_rrf", "hybrid_rerank"],
        default="bm25",
    )
```

- [ ] **Step 2: Add the `hybrid_rrf` branch**

Insert this branch in `main()` before the `else: raise ValueError(...)` (after the existing `bm25` branch):

```python
    elif args.retriever == "hybrid_rrf":
        import pickle
        from search import search
        from dense import load_dense, dense_search
        from hybrid import rrf_fuse
        with open(DATA / "index.pkl", "rb") as f:
            index = pickle.load(f)
        vindex, embedder = load_dense(DATA)
        def retrieve(q):
            bm25 = search(index, q, k=100)
            dense = dense_search(vindex, embedder, q, k=100)
            return rrf_fuse([bm25, dense])
```

- [ ] **Step 3: Add the `hybrid_rerank` branch**

Insert directly after the `hybrid_rrf` branch:

```python
    elif args.retriever == "hybrid_rerank":
        import pickle
        from search import search
        from dense import load_dense, dense_search
        from hybrid import dedup_union
        from rerank import load_doc_text, Reranker, rerank
        with open(DATA / "index.pkl", "rb") as f:
            index = pickle.load(f)
        vindex, embedder = load_dense(DATA)
        doc_text = load_doc_text(DATA)
        reranker = Reranker()
        def retrieve(q):
            bm25 = search(index, q, k=100)
            dense = dense_search(vindex, embedder, q, k=100)
            union = dedup_union([bm25, dense])
            return rerank(reranker, q, union, doc_text, k=100)
```

- [ ] **Step 4: Verify `hybrid_rrf` runs and check NDCG@10**

Run: `uv run python scripts/run_eval.py --retriever hybrid_rrf`
Expected: prints an `--- Overall ---` block with `NDCG@10=...  MRR=...  Recall@100=...`. No traceback.

- [ ] **Step 5: Verify `hybrid_rerank` runs**

Run: `uv run python scripts/run_eval.py --retriever hybrid_rerank`
Expected: same metric block, no traceback. (Loads the cross-encoder; slower.)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_eval.py
git commit -m "Add hybrid_rrf and hybrid_rerank retrievers to run_eval"
```

---

### Task 3: Benchmark report & README update

**Files:**
- Modify: `README.md` (the "Dense retrieval & cross-encoder reranking" note / retriever list)

**Interfaces:**
- Consumes: the eval CLI from Task 2.
- Produces: documented NDCG@10 numbers for the two new retrievers.

- [ ] **Step 1: Collect the head-to-head numbers**

Run each and record `NDCG@10 / MRR / Recall@100`:

```bash
uv run python scripts/run_eval.py --retriever bm25
uv run python scripts/run_eval.py --retriever dense
uv run python scripts/run_eval.py --retriever bm25_rerank
uv run python scripts/run_eval.py --retriever hybrid_rrf
uv run python scripts/run_eval.py --retriever hybrid_rerank
uv run python scripts/run_eval.py --retriever hybrid_rerank --breakdown
```

- [ ] **Step 2: Update the README note**

In `README.md`, extend the retrieval note to mention the two hybrid retrievers and add their NDCG@10 to the existing line (`BM25F 0.74, dense 0.60, BM25F+rerank 0.79`). Use the actual measured values from Step 1. Example shape (fill in real numbers):

```markdown
Evaluate any of `--retriever {bm25,dense,bm25_rerank,hybrid_rrf,hybrid_rerank}`.
`hybrid_rrf` reciprocal-rank-fuses BM25F + dense top-100; `hybrid_rerank` runs
the cross-encoder over the BM25F∪dense candidate union. On the enriched qrels
NDCG@10 is BM25F 0.74, dense 0.60, BM25F+rerank 0.79, hybrid_rrf <X>,
hybrid_rerank <Y>.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document hybrid_rrf and hybrid_rerank retrievers and their NDCG@10"
```

---

## Self-Review

**Spec coverage:**
- `hybrid_rrf` (RRF fusion, k=60, equal weights) → Task 1 (`rrf_fuse`) + Task 2 branch. ✓
- `hybrid_rerank` (union → cross-encoder) → Task 1 (`dedup_union`) + Task 2 branch reusing `rerank()`. ✓
- Tunable `k`/`weights` exposed → `rrf_fuse` signature. ✓
- Pure-unit tests, no models → `tests/test_hybrid.py` (stub-free, list math only). ✓
- Head-to-head eval vs existing four → Task 3. ✓
- No index rebuild / no new deps → reuses existing artifacts; no dependency edits. ✓

**Placeholder scan:** README `<X>`/`<Y>` are intentional fill-from-measurement slots, not plan gaps — Step 1 produces the numbers. No other placeholders.

**Type consistency:** `rrf_fuse` / `dedup_union` signatures identical between Task 1 definition and Task 2 usage. `rerank(reranker, query, candidate_ids, doc_text, k)` matches `src/rerank.py`. `search`/`dense_search` return `(doc_id, score)` lists as consumed. ✓
