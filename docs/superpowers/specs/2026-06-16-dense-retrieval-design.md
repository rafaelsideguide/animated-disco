# Dense / kNN Retrieval — Design

**Date:** 2026-06-16
**Branch:** `dense-retrieval` (off `search-improvements`)
**Sub-project:** B (relevance program: A fielded-BM25F ✓ → **B dense** → C hybrid → D cross-encoder → E router; F re-pooling deferred)

## Problem

The system is lexical (BM25F, NDCG@10 0.65). Lexical matching cannot bridge
vocabulary gaps — the paraphrase bucket is the weakest (0.44) because queries
use different words than the documents. A dense (embedding) retriever matches on
meaning, not surface form, and surfaces semantically-relevant documents that
BM25F misses. This sub-project builds a standalone dense retriever; fusing it
with BM25F is sub-project C.

## Scope decisions (agreed)

- **Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) — covers
  the ~21% non-English corpus as well as English.
- **ANN library:** **hnswlib** (light pip wheel, easy serialization, ideal at 50K;
  faiss-cpu was the alternative). Cosine space on L2-normalized vectors.
- **Embed text:** `title + headings + body-prefix` (reusing `parse.py` fields).
- **Transformers, not LLM API.**
- **Out of scope:** hybrid fusion (C), cross-encoder rerank (D), query routing
  (E), judgment re-pooling (F). Dense is a *parallel* retriever; BM25F code is
  untouched.

## Architecture

### New `src/dense.py`
Split so index logic is testable without a model download.

- `class VectorIndex` — hnswlib wrapper.
  - `build(vectors: np.ndarray, doc_ids: list[str])` — adds normalized vectors
    under integer labels; stores `doc_ids`.
  - `query(vector: np.ndarray, k: int) -> list[tuple[str, float]]` — returns
    `(doc_id, similarity)` where `similarity = 1 - cosine_distance`, descending.
  - `save(dir)` / `load(dir)` — writes `hnsw.bin` + `dense_meta.pkl`
    (`doc_ids`, `dim`, `model_name`).
  - Fully unit-testable with synthetic vectors (no torch).
- `class Embedder` — thin `SentenceTransformer` wrapper; lazy-loads the model on
  first use; `encode(texts: list[str]) -> np.ndarray` returns L2-normalized
  float32. The single place that imports torch/sentence-transformers.
- `embed_text(fields: dict) -> str` — `f"{title} {headings} {body[:BODY_EMBED_CHARS]}"`
  with `BODY_EMBED_CHARS = 512` (the model truncates at 128 tokens regardless;
  the char cap just bounds input handed to the tokenizer).
- `dense_search(vindex, embedder, query, k=10) -> list[tuple[str, float]]`.
- `load_dense(dir) -> (VectorIndex, Embedder)`.

### New `scripts/build_embeddings.py`
Offline build: stream `corpus.jsonl` → `parse_document` → `embed_text` →
`Embedder.encode` in batches → `VectorIndex.build(vectors, doc_ids)` →
`VectorIndex.save(DATA)`. Prints progress and final counts. Requires a one-time
model download (~470 MB) and network at build time.

### `scripts/run_eval.py` (refactor)
Extract the retrieval step into a retriever function and add
`--retriever {bm25,dense}` (default `bm25`). The dense branch lazily imports
`dense` so the default path never imports torch. Reused by C/D later.

### Dependencies
Add `sentence-transformers` and `hnswlib` to `pyproject.toml`
(torch comes transitively). No change to `bm25.py`, `index.py`, `search.py`.

### Artifacts
`data/hnsw.bin` (hnswlib index), `data/dense_meta.pkl` (doc_ids + dim + model name).

## Data flow

- Build: `corpus → parse_document → embed_text → Embedder.encode (batched,
  normalized) → VectorIndex.build(vectors, doc_ids) → save`
- Query: `query → Embedder.encode([q]) → VectorIndex.query(vec, k) → [(doc_id, cosine)]`

## Testing

- `tests/test_dense.py` — `VectorIndex` with synthetic vectors: nearest-neighbor
  correctness, k-truncation, cosine/similarity ordering, `save`/`load`
  round-trip, doc_id mapping. No model required.
- An `Embedder` integration test that **skips** (`unittest.SkipTest`) if the
  model cannot be loaded (offline/CI), asserting output shape `(n, 384)` and unit
  norm. Keeps the fast unit suite model-free.
- Validation: `build_embeddings.py`, then
  `run_eval.py --retriever dense --breakdown`; record the breakdown.

## Validation & success criteria

- A working dense retriever: build succeeds, eval runs, results are sensible.
- **Dense-alone NDCG@10 is expected to be BELOW BM25F's 0.65 and is a pessimistic
  lower bound** — dense surfaces semantically-relevant docs the baseline-pooled,
  title/URL-graded judgments never scored (the pool-bias issue; re-pooling is F,
  out of scope). Success is **complementarity**, not beating BM25F solo:
  - dense should be relatively strong on **paraphrase** vs its own other buckets;
  - the breakdown should show dense and BM25F have different strengths (the
    premise for C/hybrid).
- Report dense's per-bucket breakdown alongside BM25F's for comparison.

## Known limitations (deferred)

- 128-token model context: embeddings see title + headings + lead body only.
- Dense-alone is under-measured by the current qrels until re-pooling (F).
- Fusion with BM25F (C), cross-encoder rerank (D), and routing (E) are separate
  sub-projects.
