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
