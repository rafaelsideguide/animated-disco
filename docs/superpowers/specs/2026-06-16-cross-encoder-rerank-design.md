# Cross-Encoder Reranking — Design

**Date:** 2026-06-16
**Branch:** `rerank` (off `search-improvements`)
**Sub-project:** D (relevance program: A fielded-BM25F ✓ → B dense ✓ → **D rerank** → C hybrid → E router; F re-pooling deferred)

## Problem

BM25F retrieves most judged-relevant docs but ranks many of them too low: the
complementarity analysis showed BM25F recall@100 is 0.77–0.89 while recall@10 is
only 0.49–0.66 — i.e. **22–40% of relevant docs sit at ranks 11–100**, already
retrieved but not in the top-10. A cross-encoder reranker rescores BM25F's
top-100 query-doc pairs and reorders them, pulling those relevant docs up.

Because reranking only reorders BM25F's *own* retrieved (and therefore judged)
documents, the gain is **faithfully measurable on the current judgments** —
unlike dense retrieval, whose value is hidden by the baseline-pooled qrels.

## Scope decisions (agreed)

- **Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual) — scores
  non-English pairs correctly, protecting the strong non-english bucket (0.82).
- **Rerank depth:** BM25F top-100 → rerank → top-100 (eval scores NDCG@10).
- **Pure rerank:** cross-encoder score replaces BM25F's for ordering
  (score-blending is a later tuning option).
- **Doc text:** `title + headings + body[:512]` (reusing `parse.py`).
- **No new dependency:** `CrossEncoder` ships with `sentence-transformers`.
- **Out of scope:** hybrid fusion (C), query routing (E), re-pooling (F).
  BM25F/dense code untouched.

## Architecture

### New `scripts/build_doc_store.py` → `data/doc_text.pkl`
Streams `corpus.jsonl`, `parse_document(doc)` → `doc_id -> rerank_text`, where
`rerank_text = f"{title} {headings} {body[:RERANK_BODY_CHARS]}".strip()` with
`RERANK_BODY_CHARS = 512`. Pickles the dict. Uses the scripts/inspect.py
sys.path guard (numpy/torch import stdlib `inspect`).

### New `src/rerank.py`
Splits the model from the ordering logic so ordering is unit-testable.

- `RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"`
- `class Reranker` — lazy `CrossEncoder(RERANK_MODEL)`;
  `score_pairs(query: str, texts: list[str]) -> list[float]` via
  `model.predict([(query, t) for t in texts])` (returns python floats).
- `rerank(reranker, query, candidate_ids, doc_text, k=10) -> list[tuple[str, float]]`
  — builds `texts = [doc_text.get(cid, "") for cid in candidate_ids]`, scores
  them, returns the top-k `(doc_id, ce_score)` sorted by score descending.
  Missing doc_ids resolve to empty text (scored, not dropped). Pure logic.
- `load_doc_text(directory) -> dict` — loads `data/doc_text.pkl`.

### `scripts/run_eval.py`
Add `bm25_rerank` to `--retriever` choices. That branch: load BM25F index +
`load_doc_text` + `Reranker` (lazy import of `rerank`), then
`retrieve(q) = rerank(reranker, q, [d for d,_ in search(index, q, k=100)], doc_text, k=100)`.
Default and existing `bm25`/`dense` paths unchanged.

### Artifact
`data/doc_text.pkl` (LFS-tracked, like `index.pkl`/`hnsw.bin`).

## Data flow

- Build: `corpus → parse_document → {doc_id: title headings body[:512]} → data/doc_text.pkl`
- Query: `BM25F top-100 doc_ids → look up texts → CrossEncoder.predict (query, text) pairs → sort desc → top-k`

## Testing

- `tests/test_rerank.py`:
  - `rerank()` with a **stub reranker** (returns scores by lookup) — verifies
    reordering by score, top-k truncation, doc_text lookup, and that a missing
    doc_id is scored with empty text rather than crashing. No model.
  - A `Reranker` integration test that **skips** (`ImportError`/`OSError`) if the
    model is unavailable, asserting `score_pairs` returns one float per text.
  - doc-store build logic on a tiny in-memory corpus (if factored to a testable
    function) — otherwise covered by the validation run.
- Validation: build the store, run `run_eval.py --retriever bm25_rerank --breakdown`.

## Validation & success criteria

Reranking reorders BM25F's judged top-100, so the metric is trustworthy here.

- **Primary:** overall NDCG@10 improves above the 0.65 BM25F baseline.
- **Guard:** non-english (0.82) must not regress materially — the reason for the
  multilingual model. If any bucket drops, investigate doc-text length or model.
- Report the full per-bucket breakdown vs BM25F.

## Known limitations (deferred)

- Pure rerank (no BM25F+CE score blend); blending is a tuning option.
- Serve-time latency: ~100 cross-encoder passes per query (retrieve-cheap,
  rerank-expensive). Acceptable for eval; a depth/latency knob is future work.
- Doc text capped at title+headings+body[:512]; the cross-encoder sees only that.
- Hybrid (rerank over BM25F∪dense) and routing are separate sub-projects.
