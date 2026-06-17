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
multilingual cross-encoder over `data/doc_text.pkl`. Evaluate either via
`run_eval.py --retriever {dense,bm25_rerank}`.

### Re-pooled judgments

The original `judgments.jsonl` was a depth-20 pool from the BM25 baseline, which
under-measured the semantic/rerank retrievers (they surface relevant docs the
pool never judged). `judgments.jsonl` was enriched (source
`claude-code-pooled-2026`) by pooling the top-10 of BM25F + dense + rerank per
query and grading the new docs with Claude Code subagents:
`scripts/build_repool_candidates.py` → `scripts/repool_batches.py split` →
grade batches → `scripts/repool_batches.py gather` →
`scripts/merge_repool_grades.py`. Top-10 judgment coverage rose ~0.60 → 0.99; on
the enriched qrels NDCG@10 is BM25F 0.74, dense 0.60, BM25F+rerank 0.79.

AI tools (Cursor, Claude Code, etc.) are encouraged throughout the interview.

Judgments were generated with LLM assistance.
