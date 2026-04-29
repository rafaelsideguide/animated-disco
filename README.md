# Search IR Interview

Search system over 50K web pages. Eval gives NDCG@10.

## Setup

```bash
uv sync
python scripts/build_index.py
python scripts/run_eval.py
```

## Inspect the index

```bash
# Look up a term
python scripts/inspect.py term python

# Look up a document by ID
python scripts/inspect.py doc 019b76da-ae1b-7180-b9ab-2cd156dd6769

# Run a query and see ranked results with judgments
python scripts/inspect.py query q001

# Show all relevance judgments for a query
python scripts/inspect.py judgments q001
```

## Notes

AI tools (Cursor, Claude Code, etc.) are encouraged throughout the interview.

Judgments were generated with LLM assistance.
