# Tokenizer Improvements — Design

**Date:** 2026-06-16
**Branch:** `improve-tokenizer`
**Component:** `src/tokenizer.py`

## Problem

The search system is BM25 over ~50K web pages, evaluated by NDCG@10. The
tokenizer is the shared chokepoint: `tokenize()` runs at **both** index-build
time (`scripts/build_index.py`) and query time (`src/search.py`), so the index
and queries must be tokenized identically or terms won't match.

The current tokenizer is naive:

```python
STOPWORDS: set[str] = set()

def tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t not in STOPWORDS]
```

It only lowercases and splits on whitespace. No stopword list, no punctuation
handling, no hyphen splitting, no normalization.

## Baseline (NDCG@10 = 0.38 overall)

From `run_eval.py --breakdown`:

| Bucket            | NDCG@10 | MRR  | Recall@100 | n  |
|-------------------|---------|------|------------|----|
| **Overall**       | 0.38    | 0.56 | 0.45       | 197|
| natural-language  | 0.46    | 0.65 | 0.58       | 60 |
| short-keyword     | 0.48    | 0.77 | 0.48       | 40 |
| paraphrase        | 0.16    | 0.30 | 0.32       | 30 |
| hyphenated        | 0.11    | 0.19 | 0.10       | 30 |
| code-identifier   | 0.40    | 0.54 | 0.49       | 20 |
| non-english       | 0.64    | 0.90 | 0.69       | 17 |

**Weak buckets:** hyphenated (0.11) and paraphrase (0.16).
**Strong buckets to protect:** short-keyword (0.48), non-english (0.64).

### Why hyphenated fails

Queries like `treasury-yields market-commentary april-2025` tokenize as single
tokens (`treasury-yields`) under `.split()`. The corpus body uses spaces, so
`treasury-yields` matches nothing. Splitting on hyphens fixes this.

### Why paraphrase fails

Paraphrased queries use different word forms than the corpus (e.g.
`managing`/`management` vs `manage`). Exact-match tokens miss them. Stemming
collapses word forms to a common root.

### code-identifier queries

These use underscores (`guild_id`, `financial_report`, `ashby_jid`), hyphens
(`send-message`, `roto-rooter`), and alphanumeric codes (`3c7wrnfl0ng288476`,
`YmaCJxEfw`). They do **not** use `c++`/`c#`/`node.js`, so symbol-bearing
tech-token preservation is unnecessary. A Unicode `\w+` tokenizer keeps
underscores and alphanumeric codes whole while splitting hyphens — the right
behavior for this distribution.

## Design

A four-stage pipeline in `src/tokenizer.py`, applied identically at index and
query time.

1. **Normalize** — Unicode NFKD, strip combining marks (accent-fold:
   `café` → `cafe`), then `casefold()` (lowercase; handles mixed-case codes
   like `YmaCJxEfw`).
2. **Tokenize** — `re.findall(r"\w+", text)` with Unicode semantics. Splits on
   hyphens, en/em dashes, dots, and all punctuation; **keeps underscores and
   alphanumeric codes whole**; keeps CJK runs intact (protects non-english).
3. **Drop stopwords** — curated standard English list (the ~179-word NLTK list)
   embedded as the `STOPWORDS` constant. No `nltk.download` step.
4. **Stem** — Snowball/Porter2 English stemmer via the `snowballstemmer`
   library (pure-Python, no data files). **Guarded:** only stem tokens where
   `token.isalpha()` and `len(token) > 2`, so numbers, alphanumeric codes, and
   underscored identifiers pass through untouched.

### Components

- `STOPWORDS: frozenset[str]` — embedded English stopword list.
- `_normalize(text: str) -> str` — NFKD + accent strip + casefold.
- `tokenize(text: str) -> list[str]` — orchestrates the four stages.
- Module-level `snowballstemmer` English stemmer singleton.

### Dependency

Add `snowballstemmer` to `pyproject.toml`. Stopwords are embedded rather than
adding NLTK, to avoid a runtime data download.

## Data flow

`raw text → _normalize → re.findall(\w+) → drop stopwords → stem (guarded) → list[str]`

Identical at index-build and query time. Changing the tokenizer **requires
rebuilding the index** (`scripts/build_index.py`); query tokens must match
indexed tokens.

## Testing

New `tests/test_tokenizer.py` (TDD — tests first):

- hyphen / en-dash / em-dash splitting
- punctuation stripping
- accent folding (`café` → `cafe`)
- stopword removal
- stemming of word-form variants to a common root
- underscore preservation (`guild_id` stays whole)
- number / alphanumeric-code preservation (no stemming)
- edge cases: empty string, whitespace-only, all-stopwords

The existing `tests/test_smoke.py` passes tokens directly to `rank()`/the index
and bypasses `tokenize()`, so it remains valid.

## Validation

1. Rebuild the index: `uv run python scripts/build_index.py`.
2. Re-run eval: `uv run python scripts/run_eval.py --breakdown`.

**Success criteria:**

- Overall NDCG@10 improves above 0.38.
- hyphenated and paraphrase buckets improve materially.
- No bucket regresses materially — in particular non-english (0.64) and
  short-keyword (0.48).

Report the before/after breakdown table.

## Known limitations (deferred)

- Symbol-bearing tech tokens (`c++`, `c#`, `node.js`) collapse to `c` /
  `node`+`js`. Not needed by current eval queries.
- Non-English-specific tokenization (CJK segmentation, language-specific
  stemming) deferred. Current `\w+` keeps CJK runs whole, matching prior
  behavior.
- Subword dual-emission (emitting both `guild_id` and `guild`/`id`) deferred;
  could raise recall at some precision cost.
