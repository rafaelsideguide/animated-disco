# Re-pooling Judgments (Claude Code-graded) — Design

**Date:** 2026-06-16
**Branch:** `repool` (off `rerank`, which carries BM25F + dense + cross-encoder rerank)
**Sub-project:** F (relevance program: A fielded-BM25F ✓ → B dense ✓ → D rerank ✓ (built, unmeasurable on old qrels) → **F re-pool** → then C hybrid / E router)

## Problem

The judgments are a depth-~20 pool from the *original BM25 baseline*, graded by an
LLM on `URL + title + markdown[:200]` (boilerplate). Every improvement since —
dense (B) and cross-encoder rerank (D) — surfaces relevant docs that were never
in that pool, so they are unjudged and scored 0. Measured proof: rerank scores
NDCG@10 0.53 vs BM25F 0.65 while judgment coverage drops 0.60→0.39, yet
spot-checks show its promoted docs are clearly relevant (e.g. a dentist *Contact*
page for a "with phone number" query). **The qrels can no longer measure ranking
improvements.** F enriches them via TREC-style union pooling so C/D/dense become
measurable.

## Approach

Pool candidates from multiple diverse retrievers, judge the *new* (unjudged)
docs, and append to the existing judgments. Grading is done by **Claude Code
subagents** (billed to the session) rather than the metered Anthropic API, to
avoid cost. Candidate generation and merge are deterministic scripts; only the
inherently-LLM grading step is agent-driven, isolated behind two interface files.

## Scope decisions (agreed)

- **Pool:** BM25F + dense + cross-encoder-rerank, **top-10 each**, union minus
  already-judged (qid, doc_id) pairs.
- **Grader input:** cleaned text — `"URL: {url}\n{doc_text}"`, where `doc_text`
  is `title + headings + body` from D's `data/doc_text.pkl`.
- **Grader:** Claude Code grading subagents (haiku-class model — cheap, close to
  the original haiku labels), structured 0/1/2 output. No API key, no metered cost.
- **Merge:** append to `judgments.jsonl` with `source = "claude-code-pooled-2026"`;
  idempotent (skip already-judged pairs).
- **Out of scope:** re-grading the existing labels; hybrid (C); router (E).

## Architecture

### New `src/repool.py` (pure, unit-testable — no API/agent)
- `new_candidates(pools: list[list[str]], judged: set[str]) -> list[str]` — union
  of the per-system ranked lists (order-preserving, first-seen wins), dedup, drop
  any doc_id in `judged`. (`judged` is the per-query set of already-judged doc_ids.)
- `grader_doc_text(url: str, doc_text: str) -> str` — returns `f"URL: {url}\n{doc_text}".strip()`.

### New `scripts/build_repool_candidates.py` → `data/repool_candidates.json`
- sys.path guard (scripts/inspect.py shadow).
- Load queries, existing judgments (→ `{qid: set(doc_ids)}`), BM25F `index.pkl`,
  dense (`load_dense`), reranker (`Reranker`), `doc_text.pkl`.
- Per query: `bm25_100 = search(index, q, k=100)`;
  pools = `[bm25_100[:10] ids, dense top-10 ids, rerank(bm25_100) top-10 ids]`;
  `new = new_candidates(pools, judged[qid])`.
- For each new doc_id: `text = grader_doc_text(index.doc_meta[i]["url"], doc_text[doc_id])`.
- Emit a JSON list of `{"qid", "query", "docs": [{"doc_id", "text"}]}` (queries
  with no new candidates omitted). Print totals (queries, new docs).

### Agent grading step (orchestrated, not a script) → `data/repool_grades.jsonl`
- The controller dispatches grading subagents over batches of queries from
  `repool_candidates.json`. Each subagent receives a batch and returns, via a
  structured schema, a grade in {0,1,2} for every `(qid, doc_id)` it was given
  (0=not relevant, 1=somewhat, 2=highly — matching the original rubric).
- The controller writes all returned grades to `data/repool_grades.jsonl` as
  `{"qid", "doc_id", "grade"}` lines. Every candidate doc must receive a grade;
  any gaps are re-dispatched.

### New `scripts/merge_repool_grades.py`
- Loads `repool_grades.jsonl` and existing `judgments.jsonl`; appends each grade
  as `{"qid", "doc_id", "grade", "source": "claude-code-pooled-2026"}` unless that
  (qid, doc_id) is already judged. Idempotent. Prints how many added.

### Re-eval
Rerun `scripts/run_eval.py --retriever {bm25,dense,bm25_rerank} --breakdown` on
the enriched `judgments.jsonl`; report before/after NDCG and coverage per system.

## Data flow

`query → BM25F top-100 → {bm25 top-10, dense top-10, rerank top-10} → union −
already-judged → candidates.json → [grading subagents] → grades.jsonl →
merge → judgments.jsonl (enriched) → re-eval`

## Testing

- `src/repool.py`: `new_candidates` (union order, dedup, exclude judged, empty
  pools) and `grader_doc_text` (format, empty url/text). All pure, no API.
- `scripts/merge_repool_grades.py`: factor the merge into a testable function
  `merge(existing_rows, grade_rows, source) -> new_rows` and unit-test idempotency
  (already-judged skipped) and source tagging on in-memory data.
- `build_repool_candidates.py`: covered by the validation run (it composes tested
  pieces); optionally a tiny smoke that it imports.
- Validation: run the pipeline end-to-end, then re-eval.

## Validation & success criteria

This sub-project changes the *measuring stick*, so "success" is about a fairer
ruler, not a target NDCG:

- `judgments.jsonl` gains new graded (qid, doc_id) rows from the pool; judgment
  **coverage of all three retrievers' top-10 rises materially** (the old pool
  covered ~0.60 for BM25F, ~0.20–0.39 for dense/rerank).
- On the enriched qrels, **rerank (D) and dense (B) should improve relative to
  their old (under-measured) numbers**, and ideally rerank now beats or matches
  BM25F — the outcome the spot-checks predicted. Report the full before/after.
- No corruption of existing judgments (only appends; existing rows untouched).

## Honest caveats

- **Circularity:** an LLM (agent) judges docs our systems surfaced. Pooling only
  controls *which* docs get judged; each is graded independently, and pooling from
  3 diverse systems + the existing labels limits single-system bias. Not eliminated.
- **Reproducibility:** the grading step needs a Claude Code agent, not just
  `uv run`. Mitigated by the candidates/grades file interface (both inspectable)
  and a README note; the candidates can be re-graded by any method later.
- **Label heterogeneity:** new grades come from a Claude Code subagent, existing
  from the original haiku API. A haiku-class subagent keeps them close; some
  inconsistency remains.
- Labels stay depth-limited; this makes the eval **much fairer, not perfect**.
