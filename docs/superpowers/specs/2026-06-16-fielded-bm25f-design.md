# Fielded Content Representation + BM25F — Design

**Date:** 2026-06-16
**Branch:** `fielded-bm25f` (off `search-improvements`)
**Sub-project:** A (first of the relevance program: A content/fields → B dense → C hybrid → D cross-encoder → E router)

## Problem

The index is built from `title + body[:500]` concatenated into a single field
(`build_index.py:extract_text`). Two consequences:

1. **~98% of body text is absent** and the indexed 500 chars are mostly markdown
   boilerplate — skip-links, `![img](url)` tags, and bare URLs — so the body
   contributes mostly URL/filename noise tokens.
2. **No field separation.** Title terms score identically to body terms; there
   is no way to boost titles or normalize field lengths independently. The
   `title` in `doc_meta` is display-only.

This sub-project parses each document into weighted fields, cleans markdown
noise, indexes more body, and replaces single-field BM25 with **true BM25F**.

## Scope decisions (agreed)

- **True BM25F** (per-field tf combined before saturation), not term-repetition
  or per-field score summation.
- **Regex-based markdown cleaning**, no new dependency.
- **Body capped at 3000 cleaned chars** (up from 500).
- **Fields:** `title`, `headings`, `url`, `body`.
- **Out of scope:** judgment re-pooling (program step F). We proceed on the
  current (baseline-pooled, title+URL-graded) judgments, accepting that
  content/recall gains will be under-credited. Success is judged mainly on
  buckets the qrels can see (navigational, keyword, code-id, hyphenated).

## Architecture

### New module `src/parse.py` (pure functions, no deps)

- `clean_markdown(md: str) -> str`
  - remove images `![alt](url)` entirely
  - links `[text](url)` → `text`
  - remove bare URLs (`https?://…`)
  - strip heading markers, emphasis (`*` `_`), inline code backticks, blockquote
    `>`, table pipes, list markers (`- `, `* `, `1.`)
  - collapse runs of whitespace
- `extract_headings(md: str) -> str` — concatenated text of `^#{1,6}\s+…` lines.
- `url_tokens(url: str) -> str` — hostname minus leading `www.` and the TLD,
  plus the URL path, split on non-alphanumerics into space-joined tokens.
- `parse_document(doc: dict) -> dict[str, str]` — returns
  `{"title", "headings", "url", "body"}`, where `body = clean_markdown(doc["markdown"])[:3000]`.

### `src/index.py` — fielded postings (extends the current CSR layout)

`FIELDS = ("title", "headings", "url", "body")`

- `add_document(doc_id, fields: dict[str, list[str]], meta)` — `fields` maps each
  field name to its token list. Records, per (term, doc), a term frequency in
  each field, and the per-field token-count (doc length).
- Build-time: per term, `dict[doc_id] -> [tf_title, tf_headings, tf_url, tf_body]`.
- `finalize()` flattens to CSR:
  - `post_doc_ids: array` — docs containing the term in any field
  - `post_tf_title, post_tf_headings, post_tf_url, post_tf_body: array` —
    parallel per-field tfs (downcast to 2-byte where values fit, as today)
  - `post_offsets: array` — term_id → slice
  - `dl_title, dl_headings, dl_url, dl_body: list[int]` — per-field doc lengths
  - `avgdl: dict[str, float]` — per-field average length
  - `df` for a term = length of its postings slice (distinct docs, any field)
- `postings(term_id)` returns `(doc_ids, [tf arrays...], start, end)` (or a small
  struct) so `bm25`/`inspect` read per-field tfs by index.

### `src/bm25.py` — BM25F scoring

Constants (tunable):
```
K1 = 1.2   # single global saturation parameter (not per-field)
BOOST = {"title": 3.0, "url": 2.0, "headings": 1.5, "body": 1.0}
# Per-field length-normalization strength. Free-text body gets the classic ~0.75
# (long bodies match terms by chance → normalize); short, length-invariant fields
# get near-zero (title length is not a relevance signal, and title tf is ~1).
B     = {"title": 0.0, "url": 0.2, "headings": 0.4, "body": 0.75}
```
Per query term `t` with document frequency `df`:
```
idf = log((N - df + 0.5) / (df + 0.5) + 1)
for each posting doc:
    wtf = sum over fields f of:
        BOOST[f] * tf_f / (1 - B[f] + B[f] * dl_f[doc] / avgdl[f])   # skip f if tf_f == 0
    score[doc] += idf * wtf / (K1 + wtf)
ranked = nlargest(k, score.items(), key=score)     # unchanged top-k + tie-break
```

### Callers

- `scripts/build_index.py`: replace `extract_text` with `parse_document`;
  tokenize each field; call `add_document` with the per-field token dict. Vocab
  count still from `index.term_dict`.
- `scripts/inspect.py`: `term` view adapts to per-field tfs (show per-field or
  summed tf).
- `src/search.py`: unchanged — queries are tokenized once (not fielded).

## Data flow

`doc → parse_document → 4 field strings → tokenize each field → add_document
(per-field tfs + lengths) → finalize (df, per-field avgdl, CSR) → BM25F rank → top-k`

Query side: `tokenize(query)` once → BM25F over the 4 fields.

## Testing (TDD)

- `tests/test_parse.py`: image/link/URL stripping, heading extraction, url
  tokenization (host+path), 3000-char body cap, empty/whitespace edge cases.
- `tests/test_index.py`: per-field tf recorded correctly; per-field doc lengths;
  `df` counts distinct docs across fields; CSR round-trips.
- `tests/test_bm25.py` (extend): a title-only match outranks a body-only match
  at equal raw tf (boost works); per-field length normalization sanity; keep the
  `nlargest` tie-break test.
- `tests/test_smoke.py`: update to the fielded `add_document` signature.
- Integration: rebuild index, run `run_eval.py --breakdown`; record before/after.

## Validation & success criteria

The behavior-preserving `bench.py` oracle no longer applies (scoring changes by
design); re-baseline from the current branch's NDCG.

- **Primary:** overall NDCG@10 improves vs the `search-improvements` baseline
  (0.5118), driven by the faithfully-measurable buckets (navigational, keyword,
  code-id, hyphenated).
- **Watch:** if overall or a key bucket drops, investigate weights, the cleaner
  (over-stripping useful tokens), or per-field `B`. paraphrase and non-english
  gains are not expected to be well-measured here.
- Index must rebuild cleanly; all unit tests green.

## Known limitations (deferred)

- Recall/content gains under-measured until judgments are re-pooled (step F).
- No boilerplate (nav/footer) detection beyond markdown-syntax stripping; with a
  3000-char cap and length normalization the residual boilerplate is a small
  fraction.
- Field weights/`B`/`K1` use sensible defaults; systematic tuning is a later
  sub-project (relevance option C, with a train/test split).
