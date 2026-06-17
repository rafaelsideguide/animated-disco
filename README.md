# Search IR Interview

Search system over 50K web pages. Eval gives NDCG@10.

## Setup

```bash
# Install Git LFS (required for large data files)
# macOS: brew install git-lfs
# Linux: sudo apt install git-lfs  (or sudo yum install git-lfs)
# Windows: https://git-lfs.com
git lfs install

# Clone the repo
git clone https://github.com/rafaelsideguide/animated-disco.git
cd animated-disco

# Install dependencies and run eval
uv sync
uv run python scripts/build_index.py  # index is pre-built; rebuild after making changes
uv run python scripts/run_eval.py
```

## Evaluate

```bash
uv run python scripts/run_eval.py            # overall metrics
uv run python scripts/run_eval.py --breakdown  # by intent, query type, and judgment coverage
```

## Inspect the index

```bash
# Look up a term
uv run python scripts/inspect.py term python

# Look up a document by ID
uv run python scripts/inspect.py doc 019b76da-ae1b-7180-b9ab-2cd156dd6769

# Run a query and see ranked results with judgments
uv run python scripts/inspect.py query q001

# Show all relevance judgments for a query
uv run python scripts/inspect.py judgments q001
```

## Notes

### Tokenizer

`tokenize()` (in `src/tokenizer.py`) runs at both index-build and query time:
Unicode normalization + accent folding + casefold, `\w+` tokenization (splits
hyphens/dashes/punctuation, keeps underscores and alphanumeric codes whole),
English stopword removal, and guarded Snowball stemming (alphabetic tokens
longer than 2 chars). **Changing the tokenizer requires rebuilding the index**
(`scripts/build_index.py`) so indexed and query tokens stay consistent.

### Fielded BM25F

Documents are parsed into four fields — `title`, `headings`, `url` (host + path
slug), and `body` (markdown-cleaned, capped at 3000 chars) — by `src/parse.py`.
The index (`src/index.py`) stores per-field term frequencies and lengths;
`src/bm25.py` scores with BM25F, weighting fields via `BOOST` and normalizing
each field's length via per-field `B`. Changing parsing, fields, or weights
requires rebuilding the index.

### Dense retrieval & cross-encoder reranking

`src/dense.py` adds a multilingual sentence-transformer + hnswlib kNN retriever
(`scripts/build_embeddings.py`); `src/rerank.py` reranks BM25F's top-100 with a
multilingual cross-encoder over `data/doc_text.pkl`.

### Hybrid retrieval

`src/hybrid.py` combines BM25F and dense. `hybrid_rrf` reciprocal-rank-fuses
(RRF, `k=60`, equal weights) BM25F's and dense's top-100 ranked lists — no model
at query time, only ranks. `hybrid_rerank` runs the cross-encoder over the
deduplicated **union** of BM25F + dense top-100 (so dense-only candidates reach
the reranker, unlike `bm25_rerank` which only sees BM25F's pool).

Evaluate any retriever via
`run_eval.py --retriever {bm25,dense,bm25_rerank,hybrid_rrf,hybrid_rerank,router}`.

### Cost routing

`src/route.py` implements a cascade that trades off a small NDCG@10 loss for
dramatic cross-encoder savings. The `router` retriever (`--retriever router`,
tuned via `scripts/tune_router.py`) runs BM25F first, then uses a confidence
gate to decide whether to return BM25F alone or escalate to a more costly model.
If the margin between BM25F's top score and its k-th score (normalized:
`(s₀ − sₖ)/s₀`) meets the threshold `TAU = 0.30`, the query is served by BM25F
with no further ranking. Otherwise, the router escalates: non-ASCII queries
route to `hybrid_rrf` (dense + BM25F fusion, no cross-encoder), all others to
`bm25_rerank` (cross-encoder reranking of BM25F's top-100).

On the re-pooled qrels (n=200), the router achieves **NDCG@10 0.74** (within
0.01 of `bm25_rerank`'s 0.75) while invoking the cross-encoder on only **108 /
200 queries** (~46% fewer calls than always-`bm25_rerank`). Cross-encoder
invocations cost ~10,800 query-doc pair scores vs ~20,000 for always-reranking.
Tier breakdown: 88 queries served by BM25F alone, 4 by hybrid_rrf, 108 by
bm25_rerank.

**Caveat:** `TAU` is tuned and reported on the same 200 queries with no
train/test holdout, so this number is a near-upper-bound, not a held-out
estimate. Further tuning on a held-out set may differ.

### Re-pooled judgments

The original `judgments.jsonl` was a depth-20 pool from the BM25 baseline, which
under-measured retrievers that surface relevant docs the pool never judged.
`judgments.jsonl` is enriched (source `claude-code-pooled-2026`) by pooling the
**top-25 of all five retrievers** — BM25F, dense, rerank, hybrid_rrf,
hybrid_rerank — per query and grading the new docs with Claude Code subagents:
`scripts/build_repool_candidates.py` → `scripts/repool_batches.py split` →
grade batches → `scripts/repool_batches.py gather` →
`scripts/merge_repool_grades.py`. (The pool started at the top-10 of BM25F +
dense + rerank; it was deepened to top-25 and extended to the two hybrids so
every retriever's top-10 is judged on equal footing.) Top-10 judgment coverage
is ~1.00 for all five.

On the re-pooled qrels (n=200), NDCG@10 / MRR / Recall@100:

| retriever      | NDCG@10 | MRR  | Recall@100 |
|----------------|:-------:|:----:|:----------:|
| bm25 (BM25F)   |  0.70   | 0.96 |    0.76    |
| dense          |  0.62   | 0.86 |    0.57    |
| bm25_rerank    |  0.75   | 0.96 |    0.76    |
| hybrid_rrf     |  0.71   | 0.93 |    0.87    |
| hybrid_rerank  |  0.72   | 0.94 |    0.82    |

`bm25_rerank` leads on NDCG@10. Both hybrids beat plain BM25F and dense, but
adding dense candidates to the reranker (`hybrid_rerank`) does not overtake
reranking BM25F alone on top-10 precision. The hybrids' clear win is
**Recall@100** (0.87 / 0.82 vs 0.76) — fusing dense with BM25F surfaces more
relevant docs deep in the list, even when it doesn't reorder the top-10 better.

AI tools (Cursor, Claude Code, etc.) are encouraged throughout the interview.

Judgments were generated with LLM assistance.
