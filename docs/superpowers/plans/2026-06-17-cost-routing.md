# Cost-Routing Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route each query to the cheapest retriever that suffices — BM25F when confident, else dense-fusion (non-ASCII) or cross-encoder rerank — minimizing cross-encoder calls while holding NDCG@10 ≥ 0.74.

**Architecture:** A new `src/route.py` holds a confidence metric (`bm25_margin`), a non-ASCII escalation rule (`escalation_target`), and a `CostRouter` that runs the cascade over injected sub-retriever callables and tallies cost counters. `scripts/tune_router.py` sweeps the gate threshold τ; `run_eval.py --retriever router` runs it and prints a cost summary.

**Tech Stack:** Python 3.11+, pytest (unittest style), existing `index.pkl` / dense index / `doc_text.pkl`. No new dependencies, no index rebuild.

## Global Constraints

- `requires-python = ">=3.11"`; no new dependencies.
- Tests are `unittest` classes under pytest, importing `src/` via `sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))` (match `tests/test_rerank.py`).
- Retrieve functions return `list[tuple[str, float]]` `(doc_id, score)` sorted best-first.
- Routing policy: run BM25F top-100 first; if `bm25_margin(scored) >= TAU` return BM25F (skip all models); else escalate — non-ASCII query → `hybrid_rrf` (dense + RRF, no cross-encoder), otherwise → `bm25_rerank` (cross-encoder over BM25F top-100).
- `bm25_margin(scored, k=10) = (s0 - s_k) / s0`, `s_k` = score at rank `k` (or last available); returns `0.0` for empty/degenerate/`s0<=0` input (forces escalation).
- Cost metric = cross-encoder invocations + query-doc pairs scored. NDCG@10 floor = 0.74 (within 0.01 of always-`bm25_rerank` = 0.75).
- `CostRouter` takes **injected callables** (`bm25_fn`, `dense_fn`, `rerank_fn`) so it is testable without models.

---

### Task 1: `src/route.py` — confidence metric, escalation rule, CostRouter

**Files:**
- Create: `src/route.py`
- Test: `tests/test_route.py`

**Interfaces:**
- Consumes: `rrf_fuse` from `hybrid` (Task already merged).
- Produces:
  - `bm25_margin(scored: list[tuple[str, float]], k: int = 10) -> float`
  - `escalation_target(query: str) -> str` (`"hybrid_rrf"` | `"bm25_rerank"`)
  - `TAU: float` — module constant, gate threshold (retuned in Task 2).
  - `CostRouter(bm25_fn, dense_fn, rerank_fn, tau)` with `.retrieve(query, k=100) -> list[tuple[str, float]]` and `.stats: dict`. Callable contracts: `bm25_fn(query) -> [(id, score)]` (top-100); `dense_fn(query) -> [(id, score)]` (top-100); `rerank_fn(query, candidate_ids: list[str]) -> [(id, score)]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_route.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_route.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'route'`.

- [ ] **Step 3: Write the implementation**

Create `src/route.py`:

```python
# Cost-routing retriever: cheap BM25F first, escalate only when uncertain.
# TAU is the BM25F-confidence gate threshold; retune with scripts/tune_router.py.

from hybrid import rrf_fuse

TAU = 0.30


def bm25_margin(scored, k=10):
    """Normalized top-score margin (s0 - s_k) / s0 over the top-k BM25F scores.

    High margin = a clear top result = BM25F is confident. Returns 0.0 for
    empty/single-result/non-positive-top input (forces escalation)."""
    if not scored:
        return 0.0
    s0 = scored[0][1]
    if s0 <= 0 or len(scored) < 2:
        return 0.0
    idx = min(k, len(scored) - 1)
    sk = scored[idx][1]
    return (s0 - sk) / s0


def escalation_target(query):
    """Pick the escalation retriever: non-ASCII queries (≈ non-English) go to
    dense fusion (no cross-encoder); everything else to cross-encoder rerank."""
    if any(ord(c) > 127 for c in query):
        return "hybrid_rrf"
    return "bm25_rerank"


class CostRouter:
    """Routes each query to the cheapest sufficient retriever and tallies cost.

    Sub-retrievers are injected callables (so this is testable without models):
      bm25_fn(query)   -> [(id, score)]  BM25F top-100
      dense_fn(query)  -> [(id, score)]  dense top-100
      rerank_fn(query, candidate_ids) -> [(id, score)]  cross-encoder rerank
    """

    def __init__(self, bm25_fn, dense_fn, rerank_fn, tau=TAU):
        self.bm25_fn = bm25_fn
        self.dense_fn = dense_fn
        self.rerank_fn = rerank_fn
        self.tau = tau
        self.stats = {
            "bm25_only": 0,
            "hybrid_rrf": 0,
            "bm25_rerank": 0,
            "cross_encoder_calls": 0,
            "pairs_scored": 0,
        }

    def retrieve(self, query, k=100):
        scored = self.bm25_fn(query)
        if bm25_margin(scored) >= self.tau:
            self.stats["bm25_only"] += 1
            return scored[:k]
        if escalation_target(query) == "hybrid_rrf":
            dense = self.dense_fn(query)
            self.stats["hybrid_rrf"] += 1
            return rrf_fuse([scored, dense])[:k]
        candidates = [d for d, _ in scored]
        self.stats["bm25_rerank"] += 1
        self.stats["cross_encoder_calls"] += 1
        self.stats["pairs_scored"] += len(candidates)
        return self.rerank_fn(query, candidates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_route.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/route.py tests/test_route.py
git commit -m "Add cost-routing retriever: bm25_margin gate + CostRouter cascade"
```

---

### Task 2: `scripts/tune_router.py` — sweep τ, set the tuned threshold

**Files:**
- Create: `scripts/tune_router.py`
- Modify: `src/route.py` (the `TAU` constant — set to the swept value)

**Interfaces:**
- Consumes: `search`, `load_dense`/`dense_search`, `rerank`/`Reranker`/`load_doc_text`, `rrf_fuse`, `bm25_margin`, `escalation_target`; `eval as eval_module`.
- Produces: the chosen `TAU` value (printed + written into `src/route.py`).

- [ ] **Step 1: Write the tuning script**

Create `scripts/tune_router.py`:

```python
import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
import pathlib

import eval as eval_module
from search import search
from dense import load_dense, dense_search
from rerank import load_doc_text, Reranker, rerank
from hybrid import rrf_fuse
from route import bm25_margin, escalation_target

DATA = pathlib.Path(__file__).parent.parent / "data"
FLOOR = 0.74
TAUS = [i / 100 for i in range(0, 101, 5)]  # 0.00 .. 1.00 step 0.05


def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def load_judgments(path):
    j = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            j.setdefault(r["qid"], {})[r["doc_id"]] = r["grade"]
    return j


def main():
    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)
    vindex, embedder = load_dense(DATA)
    doc_text = load_doc_text(DATA)
    reranker = Reranker()

    # Precompute per query ONCE: BM25F ranking + margin, and the escalated
    # ranking (rerank or hybrid_rrf). Sweeping tau then only re-picks bm25 vs
    # escalated — no retriever re-runs.
    per_q = {}
    for qid, row in queries.items():
        q = row["query"]
        scored = search(index, q, k=100)
        target = escalation_target(q)
        if target == "hybrid_rrf":
            dense = dense_search(vindex, embedder, q, k=100)
            escalated = [d for d, _ in rrf_fuse([scored, dense])[:100]]
            uses_xenc = False
        else:
            candidates = [d for d, _ in scored]
            escalated = [d for d, _ in rerank(reranker, q, candidates, doc_text, k=100)]
            uses_xenc = True
        per_q[qid] = {
            "bm25": [d for d, _ in scored],
            "margin": bm25_margin(scored),
            "escalated": escalated,
            "uses_xenc": uses_xenc,
        }

    print(f"{'tau':>6} {'NDCG@10':>8} {'xenc_calls':>11} {'bm25_only':>10}")
    frontier = []
    for tau in TAUS:
        results, calls, bm25_only = {}, 0, 0
        for qid, d in per_q.items():
            if d["margin"] >= tau:
                results[qid] = d["bm25"]
                bm25_only += 1
            else:
                results[qid] = d["escalated"]
                if d["uses_xenc"]:
                    calls += 1
        ndcg = eval_module.evaluate(results, judgments)["ndcg@10"]
        frontier.append((tau, ndcg, calls))
        print(f"{tau:>6.2f} {ndcg:>8.3f} {calls:>11} {bm25_only:>10}")

    # Recommend: fewest cross-encoder calls subject to NDCG@10 >= FLOOR
    # (tie-break: lowest tau).
    eligible = [(calls, tau, ndcg) for tau, ndcg, calls in frontier if ndcg >= FLOOR]
    if eligible:
        calls, tau, ndcg = min(eligible)
        print(f"\nRecommended TAU={tau:.2f}  (NDCG@10={ndcg:.3f}, xenc_calls={calls}, floor={FLOOR})")
    else:
        print(f"\nNo tau meets the NDCG@10 floor of {FLOOR}; inspect the frontier above.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sweep**

Run: `uv run python scripts/tune_router.py`
Expected: a τ/NDCG/calls table, then a `Recommended TAU=<value> (...)` line with
NDCG@10 ≥ 0.74. Record the recommended τ.

- [ ] **Step 3: Set `TAU` in `src/route.py` to the recommended value**

Edit the `TAU = 0.30` line in `src/route.py`, replacing `0.30` with the recommended
value from Step 2, and update the comment to note it. Example (use the real value):

```python
# TAU is the BM25F-confidence gate threshold; tuned via scripts/tune_router.py
# to the fewest-cross-encoder-calls value holding NDCG@10 >= 0.74.
TAU = 0.15
```

- [ ] **Step 4: Confirm tests still pass**

Run: `uv run pytest tests/test_route.py -q`
Expected: PASS (tests pass `tau` explicitly, so they are independent of `TAU`).

- [ ] **Step 5: Commit**

```bash
git add scripts/tune_router.py src/route.py
git commit -m "Tune router gate threshold via tau sweep (NDCG@10 floor 0.74)"
```

---

### Task 3: Wire `--retriever router` into `run_eval.py` with a cost summary

**Files:**
- Modify: `scripts/run_eval.py` (the `--retriever` `choices`, the if/elif chain, and the end of `main()`)

**Interfaces:**
- Consumes: `CostRouter`, `TAU` from `route`; `search`, `load_dense`/`dense_search`, `rerank`/`Reranker`/`load_doc_text`.
- Produces: a `router` CLI retriever plus a printed cost summary.

- [ ] **Step 1: Add `router` to the argparse choices**

In `scripts/run_eval.py`, change the `--retriever` choices:

```python
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "bm25_rerank", "hybrid_rrf", "hybrid_rerank", "router"],
        default="bm25",
    )
```

- [ ] **Step 2: Add the `router` branch**

Insert this branch immediately before the final `else: raise ValueError(...)` in `main()`:

```python
    elif args.retriever == "router":
        import pickle
        from search import search
        from dense import load_dense, dense_search
        from rerank import load_doc_text, Reranker, rerank
        from route import CostRouter, TAU
        with open(DATA / "index.pkl", "rb") as f:
            index = pickle.load(f)
        vindex, embedder = load_dense(DATA)
        doc_text = load_doc_text(DATA)
        reranker = Reranker()
        router = CostRouter(
            bm25_fn=lambda q: search(index, q, k=100),
            dense_fn=lambda q: dense_search(vindex, embedder, q, k=100),
            rerank_fn=lambda q, cand: rerank(reranker, q, cand, doc_text, k=100),
            tau=TAU,
        )
        retrieve = router.retrieve
```

- [ ] **Step 3: Print the cost summary at the end of `main()`**

At the very end of `main()` (after the `if args.breakdown:` block), append:

```python
    if args.retriever == "router":
        s = router.stats
        n = sum(s[t] for t in ("bm25_only", "hybrid_rrf", "bm25_rerank"))
        print(f"\n--- Router cost (tau={router.tau}) ---")
        print(f"Tier usage:  bm25_only={s['bm25_only']}  "
              f"hybrid_rrf={s['hybrid_rrf']}  bm25_rerank={s['bm25_rerank']}  (n={n})")
        print(f"Cross-encoder calls:  {s['cross_encoder_calls']}/{n}  "
              f"(always-rerank would be {n})")
        print(f"Query-doc pairs scored:  {s['pairs_scored']:,}")
```

- [ ] **Step 4: Verify the router runs**

Run: `uv run python scripts/run_eval.py --retriever router --breakdown`
Expected: an `--- Overall ---` metric block with NDCG@10 ≥ 0.74, the per-intent /
per-type / coverage breakdown, then a `--- Router cost ---` block showing tier
usage and `cross_encoder_calls` well below `n` (the cost saving). No traceback.
Record NDCG@10 / MRR / Recall@100 and the cross-encoder call count.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py
git commit -m "Add router retriever to run_eval with cost summary"
```

---

### Task 4: Document the cost router in the README

**Files:**
- Modify: `README.md` (the retrieval notes)

- [ ] **Step 1: Collect the numbers** from Task 3's run (NDCG@10, MRR, Recall@100,
  tier usage, cross-encoder calls vs n).

- [ ] **Step 2: Add a "Cost routing" note** after the "Hybrid retrieval" section,
  describing the cascade (BM25F → confidence gate `TAU` → non-ASCII→hybrid_rrf /
  else→bm25_rerank), adding `router` to the `--retriever` list, and reporting the
  **measured** NDCG@10 alongside the cross-encoder-call saving (e.g. "reranks
  only X/200 queries vs 200, at NDCG@10 Y"). Include the honest caveat: `TAU` is
  tuned and reported on the same 200 queries (no train/test holdout), so the
  number is a near-upper-bound, not a held-out estimate. Use real values.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document cost-routing retriever and its cross-encoder savings"
```

---

## Self-Review

**Spec coverage:**
- BM25F-first cascade + confidence gate → Task 1 (`bm25_margin`, `CostRouter`). ✓
- Non-ASCII → hybrid_rrf / else → bm25_rerank escalation → Task 1 (`escalation_target`, `CostRouter.retrieve`). ✓
- τ tuning to NDCG@10 floor 0.74, minimizing cross-encoder calls → Task 2 (`tune_router.py`). ✓
- Cost instrumentation (tier counts, cross-encoder calls, pairs scored) → Task 1 (`stats`) + Task 3 (summary print). ✓
- `--retriever router` eval path → Task 3. ✓
- Pure-unit tests with stubs, no models → Task 1 (`tests/test_route.py`). ✓
- README report + overfitting caveat → Task 4. ✓
- No new deps / no index rebuild → reuses existing artifacts. ✓

**Placeholder scan:** `TAU = 0.30` in Task 1 is a real working default (tests pass `tau` explicitly); Task 2 retunes it to the swept value. The `TAU = 0.15` in Task 2 Step 3 is an illustrative example explicitly marked "use the real value". README values in Task 4 are fill-from-measurement (Step 1 produces them). No forbidden placeholders.

**Type consistency:** `bm25_margin(scored, k)`, `escalation_target(query)`, and `CostRouter(bm25_fn, dense_fn, rerank_fn, tau)` signatures identical across Task 1 definition, Task 2 usage, and Task 3 wiring. `rerank_fn(query, candidate_ids)` matches the `lambda q, cand: rerank(...)` wiring. `rrf_fuse` / `search` / `dense_search` / `rerank` return `(doc_id, score)` lists as consumed. `router.stats` keys match between Task 1 and Task 3's print. ✓
