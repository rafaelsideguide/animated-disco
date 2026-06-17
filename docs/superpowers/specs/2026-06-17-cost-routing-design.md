# Cost-Routing Retriever — Design

**Date:** 2026-06-17
**Status:** Approved

## Goal

Cut retrieval cost — specifically cross-encoder invocations, the dominant
expense — by routing each query to the cheapest retriever that suffices, instead
of always running the reranker. Optimize: **minimize cross-encoder calls subject
to a hard NDCG@10 floor** of 0.74 (within 0.01 of always-`bm25_rerank` = 0.75 on
the re-pooled qrels).

## Motivation

Per-bucket results (re-pooled qrels) show reranking does not always pay:

| bucket | BM25F | bm25_rerank | reranking helps? |
|--------|:-----:|:-----------:|------------------|
| keyword | 0.78 | 0.77 | no — BM25F already wins |
| natural | 0.70 | 0.75 | yes |
| paraphrase | 0.66 | 0.73 | yes |
| hyphenated | 0.74 | 0.83 | yes |
| code_id | 0.62 | 0.79 (hybrid_rerank) | yes |
| non_english | 0.60 | 0.68 (hybrid_rrf) | yes (via dense, no cross-encoder) |

The gold `query_type` labels are **not** reliably recoverable from query text
(code_id queries read like normal keyword queries; non_english is mostly
Latin-script). So routing uses **surface signals that predict "BM25F is already
enough"**, not a reconstruction of `query_type`:

- **English stopwords** separate natural/paraphrase (have them, reranking helps)
  from keyword/hyphenated/code_id/non_english (~zero stopwords).
- **Zero-stopword does not uniformly mean BM25F suffices** — keyword yes, but
  hyphenated/code_id still benefit. A **BM25F score-confidence gate** makes that
  call empirically.
- **Non-ASCII** characters catch ~half of non_english (accented/Arabic), routed
  to dense fusion (no cross-encoder).

## Architecture

New module `src/route.py`, exposed as a `--retriever router` choice in
`scripts/run_eval.py`. Reuses existing retrievers and artifacts (`index.pkl`,
dense index, `doc_text.pkl`); no index rebuild, no new dependencies.

### Routing policy (cost cascade)

Per query:

1. **Always run BM25F first** (cheapest; no model).
2. **Score-confidence gate** — compute a normalized top-margin from BM25F's
   `(id, score)` list. If `margin >= τ`, return BM25F's top-10 and stop — no
   dense, no cross-encoder. The universal cost lever.
3. **Escalate** (only when not confident) to a rule-chosen target:
   - query contains **non-ASCII** chars → `hybrid_rrf` (BM25F + dense + RRF;
     **no cross-encoder**), the best bucket for non-English and cheaper than
     reranking;
   - otherwise → `bm25_rerank` (cross-encoder over BM25F's top-100).

The **rule** picks the escalation *target*; the **gate** decides *whether* to
escalate. The cross-encoder runs only for non-confident, ASCII queries.

### Components in `src/route.py`

- `bm25_margin(scored: list[tuple[str, float]], k: int = 10) -> float` —
  confidence metric `(s_0 - s_k) / s_0` over the top-k BM25F scores; `s_k` is the
  score at rank `k` (or the last available). Returns 0.0 for empty/degenerate
  input (forces escalation).
- `escalation_target(query: str) -> str` — returns `"hybrid_rrf"` if the query
  has any non-ASCII char, else `"bm25_rerank"`.
- `TAU: float` — the tuned gate threshold, stored as a module constant (like
  `bm25.py`'s hand-tuned `BOOST`/`B`).
- `CostRouter` — holds the sub-retrievers (BM25F `search`, dense `VectorIndex` +
  `Embedder`, `Reranker` + `doc_text`) and counters; `.retrieve(query, k=100) ->
  list[tuple[str, float]]` runs the cascade. Counters: per-tier counts
  (`bm25_only`, `hybrid_rrf`, `bm25_rerank`), `cross_encoder_calls`, and
  `pairs_scored` (query-doc pairs sent to the cross-encoder).

## Data flow

```
query → BM25F top-100 ─→ margin ≥ τ ? ─yes→ BM25F top-10            (tier: bm25_only)
                                   └─no→ non-ASCII ? ─yes→ hybrid_rrf (tier: hybrid_rrf,  no x-enc)
                                                     └─no→ bm25_rerank (tier: bm25_rerank, x-enc)
```

## Threshold tuning

`scripts/tune_router.py` sweeps `τ` over candidate BM25F-margin values on the
200-query eval, printing the `(τ, NDCG@10, cross_encoder_calls)` frontier. The
selected `τ` is the **most aggressive** (lowest cost) value keeping NDCG@10 ≥
0.74. That value is written into `route.TAU`.

**Caveat (stated in README/spec):** `τ` is tuned and reported on the same 200
queries — overfitting risk on a small set, no train/test holdout. The reported
number is a near-upper-bound for this policy, not a held-out estimate.

## Instrumentation & eval

The `router` branch in `run_eval.py` builds a `CostRouter`, runs the eval, then
prints a cost summary after the metrics:

- tier distribution (bm25_only / hybrid_rrf / bm25_rerank counts),
- **cross-encoder invocations** and **total query-doc pairs scored**, vs the
  always-`bm25_rerank` baseline (200 calls / ~20k pairs).

`run_eval.py --retriever router --breakdown` reports NDCG@10 / MRR / Recall@100
plus the cost summary.

## Testing

`tests/test_route.py`, pure unit tests with stubs (no models loaded):

- `bm25_margin` — known score lists → expected margin; empty and single-result
  edge cases return 0.0.
- `escalation_target` — ASCII query → `"bm25_rerank"`; accented/Arabic query →
  `"hybrid_rrf"`.
- `CostRouter.retrieve` with stub sub-retrievers:
  - confident BM25F (margin ≥ τ) → returns BM25F results; **cross-encoder not
    called** (assert stub reranker call count == 0); tier `bm25_only`.
  - non-confident ASCII → reranker called once; tier `bm25_rerank`.
  - non-confident non-ASCII → dense/RRF path; reranker not called; tier
    `hybrid_rrf`.
  - counters (`cross_encoder_calls`, `pairs_scored`, tier counts) tally
    correctly across a mixed sequence.

Plus a manual `run_eval.py --retriever router --breakdown` run reporting NDCG@10
and the cost summary.

## Out of scope (YAGNI)

- Learned query classifier (gold `query_type` is not surface-recoverable anyway).
- Per-bucket / per-query adaptive `τ`.
- Latency / wall-clock benchmarking — cross-encoder call count is the cost proxy.
- Routing optimized for Recall@100 rather than NDCG@10.
- Train/test split for `τ` (dataset too small; caveat instead).
