# Hybrid Retriever — Design

**Date:** 2026-06-17
**Status:** Approved

## Goal

Combine the existing BM25F and dense retrievers into hybrid retrievers and
measure them head-to-head against the current `bm25`, `dense`, and `bm25_rerank`
on the enriched qrels (NDCG@10 / MRR / Recall@100). The motivation: the current
`bm25_rerank` reranks **only** BM25F's top-100, so relevant docs that dense
surfaces but BM25F misses never reach the cross-encoder. A hybrid closes that
gap. We implement two distinct approaches and let the eval decide which wins.

## Two new retrievers

Both are added as selectable `--retriever` choices in `scripts/run_eval.py`,
following the existing pattern. Neither requires an index rebuild — they reuse
`data/index.pkl`, `data/hnsw.bin` + `data/dense_meta.pkl`, and `data/doc_text.pkl`.

- **`hybrid_rrf`** — Reciprocal-rank fusion of BM25F top-100 and dense top-100.
  No model at query time; pure rank fusion. Equal weights, `k=60`.
- **`hybrid_rerank`** — Cross-encoder over the **deduplicated union** of BM25F
  top-100 + dense top-100 (up to ~200 candidates), via the existing `rerank()`.
  This is "hybrid candidate generation feeding the reranker we already built."

These compose cleanly: `hybrid_rerank` is the union pool → existing reranker;
`hybrid_rrf` is the model-free fusion baseline.

## Architecture & components

### New module: `src/hybrid.py`

```python
def rrf_fuse(ranked_lists: list[list[tuple[str, float]]],
             k: int = 60,
             weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion.

    Score(d) = Σ_i w_i / (k + rank_i(d)), where rank_i is the 0-based rank of d
    in list i (lists already sorted best-first). A doc absent from a list
    contributes nothing from that list. The per-list scores are IGNORED — only
    ranks matter, which is what makes RRF robust to incomparable score scales
    (BM25F magnitudes vs. cosine similarity).

    Returns docs sorted by fused score descending. `weights` defaults to all-1.0;
    a length mismatch with `ranked_lists` raises ValueError.
    """
```

A small `dedup_union(ranked_lists) -> list[str]` helper (first-seen order
preserved) builds the candidate pool for `hybrid_rerank`. Kept in `hybrid.py`
next to the fusion logic.

### Reused, unchanged

- `src/search.py` `search()` → BM25F top-100.
- `src/dense.py` `dense_search()` → dense top-100.
- `src/rerank.py` `rerank()` → cross-encoder reorder of a candidate id list.
  `hybrid_rerank` calls this with the union pool; no new model code.

### Eval wiring: `scripts/run_eval.py`

Extend `--retriever` choices to add `hybrid_rrf` and `hybrid_rerank`.

- `hybrid_rrf` branch: load `index.pkl` + `load_dense(DATA)`; `retrieve(q)`
  returns `rrf_fuse([search(...), dense_search(...)])`.
- `hybrid_rerank` branch: additionally load `doc_text` + `Reranker()` (mirrors
  the existing `bm25_rerank` branch); `retrieve(q)` builds the union pool and
  calls `rerank(reranker, q, union, doc_text, k=100)`.

Each `retrieve(q)` returns the standard `list[(doc_id, score)]`, so the rest of
`run_eval.py` — `--breakdown`, intent/query-type groups, judgment coverage — is
untouched.

## Data flow

```
query ─┬─ search(index, q, k=100) ───────► bm25 list  ─┐
       └─ dense_search(vindex, q, k=100) ─► dense list ─┤
                                                        ├─► hybrid_rrf:    rrf_fuse([bm25, dense])        → top-100
                                                        └─► hybrid_rerank: rerank(union(bm25, dense))     → top-100
```

## Defaults & rationale

- **RRF `k=60`** — canonical Cormack et al. (2009) value.
- **Equal weights** — the honest neutral baseline for a "which wins" comparison.
- `k` and `weights` are exposed as parameters for later tuning, but the eval
  uses the neutral defaults. No grid search in this scope.

## Testing

`tests/test_hybrid.py`, pure unit tests, **no models loaded**:

- **RRF math** — known ranked lists → hand-computed fused scores
  (e.g. `1/(60+0) + 1/(60+1)`).
- **RRF ordering property** — a doc ranked moderately in *both* lists beats a
  doc ranked #1 in only one list.
- **Dedup/union** — overlapping ids collapse; first-seen order preserved;
  union size correct.
- **Edge cases** — empty list; single list; `weights` length mismatch raises
  `ValueError`.

Plus a manual eval run: `run_eval.py --retriever hybrid_rrf` and
`--retriever hybrid_rerank` (with `--breakdown`), reporting NDCG@10 / MRR /
Recall@100 against the existing four retrievers for the head-to-head.

## Out of scope (YAGNI)

- Weight tuning / grid search over `k` and `weights`.
- New judgment re-pooling — the enriched qrels already pool BM25F + dense +
  rerank top-10, so the hybrid candidates are well-covered at depth 10.
- Query-time caching of embeddings or scores.
