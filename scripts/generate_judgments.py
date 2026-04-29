"""
Generate relevance judgments for search/IR evaluation.

Strategy:
  For each query, retrieve top-100 docs via the inverted index, take the
  top-20 as candidates, and ask Claude to grade each one 0/1/2.
"""

import sys
import os

# Remove scripts/ from sys.path to prevent scripts/inspect.py shadowing stdlib inspect
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
import time
import pathlib
import re

import anthropic
from dotenv import load_dotenv

from search import search

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"

CORPUS_PATH = DATA / "corpus.jsonl"
INDEX_PATH = DATA / "index.pkl"
QUERIES_PATH = DATA / "queries.jsonl"
JUDGMENTS_PATH = DATA / "judgments.jsonl"

SOURCE = "llm-assisted-2025-Q4"
CANDIDATES_PER_QUERY = 20
SEARCH_K = 100
MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_corpus(path: pathlib.Path) -> dict:
    """Return {doc_id: doc_dict} for fast lookup."""
    corpus = {}
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["id"]] = doc
    return corpus


def load_index(path: pathlib.Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_queries(path: pathlib.Path) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_judged_qids(path: pathlib.Path) -> set:
    """Return set of qids that already have at least one judgment (for resume)."""
    judged = set()
    if not path.exists():
        return judged
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            judged.add(row["qid"])
    return judged


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def build_prompt(query: str, candidates: list[dict]) -> str:
    """
    Build the grading prompt.

    IMPORTANT: URL is shown BEFORE title and content.
    """
    doc_blocks = []
    for i, doc in enumerate(candidates, start=1):
        url = doc.get("url", "")
        title = doc.get("title", "")
        markdown = doc.get("markdown", "")
        snippet = markdown[:200]
        doc_blocks.append(
            f"DOC {i}:\n"
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Content: {snippet}"
        )

    docs_section = "\n\n".join(doc_blocks)

    return (
        "Rate the relevance of each document to the query. "
        "Grade: 2=highly relevant, 1=somewhat relevant, 0=not relevant.\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{docs_section}\n\n"
        "Output one line per document: \"DOC {i}: {grade}\"\n"
        "No explanation needed."
    )


def parse_grades(response_text: str, n_docs: int) -> list[int | None]:
    """Parse 'DOC i: g' lines from the model response into a list of grades."""
    grades: list[int | None] = [None] * n_docs
    pattern = re.compile(r"DOC\s+(\d+)\s*:\s*([012])")
    for match in pattern.finditer(response_text):
        idx = int(match.group(1)) - 1  # convert 1-based to 0-based
        grade = int(match.group(2))
        if 0 <= idx < n_docs:
            grades[idx] = grade
    return grades


def grade_candidates(
    client: anthropic.Anthropic,
    query: str,
    candidates: list[dict],
) -> list[int | None]:
    """Call Claude once to grade all candidates; retry once on error."""
    prompt = build_prompt(query, candidates)

    def call_api():
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    try:
        text = call_api()
    except Exception as exc:
        print(f"  API error: {exc}. Retrying in 5s...")
        time.sleep(5)
        try:
            text = call_api()
        except Exception as exc2:
            print(f"  Retry failed: {exc2}. Skipping query.")
            return [None] * len(candidates)

    return parse_grades(text, len(candidates))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()  # searches cwd and parent dirs; also picks up shell env vars

    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)
    print(f"  {len(corpus):,} documents loaded.")

    print("Loading index...")
    index = load_index(INDEX_PATH)

    print("Loading queries...")
    queries = load_queries(QUERIES_PATH)
    print(f"  {len(queries)} queries loaded.")

    judged_qids = load_judged_qids(JUDGMENTS_PATH)
    if judged_qids:
        print(f"  Resuming: {len(judged_qids)} queries already judged, skipping.")

    client = anthropic.Anthropic()

    n_total = len(queries)
    written = 0

    with open(JUDGMENTS_PATH, "a") as out_f:
        for qi, q in enumerate(queries, start=1):
            qid = q["qid"]
            query_text = q["query"]

            if qid in judged_qids:
                continue

            print(f"Query {qid} ({qi}/{n_total}): {query_text!r}")

            # 1. Retrieve top-100, take top-20 as candidates
            results = search(index, query_text, k=SEARCH_K)
            candidate_ids = [doc_id for doc_id, _score in results[:CANDIDATES_PER_QUERY]]

            # 2. Filter to docs that exist in corpus
            candidates = [corpus[doc_id] for doc_id in candidate_ids if doc_id in corpus]

            if not candidates:
                print("  No corpus-matched candidates, skipping.")
                continue

            # 3. Grade via Claude
            grades = grade_candidates(client, query_text, candidates)

            # 4. Write judgments immediately
            for doc, grade in zip(candidates, grades):
                if grade is None:
                    continue  # skip unparseable grades
                record = {
                    "qid": qid,
                    "doc_id": doc["id"],
                    "grade": grade,
                    "source": SOURCE,
                }
                out_f.write(json.dumps(record) + "\n")
                written += 1
            out_f.flush()

            print(f"  Wrote {sum(g is not None for g in grades)} judgments.")

            time.sleep(0.5)

    print(f"\nDone. Total judgments written this run: {written}")
    print(f"Output: {JUDGMENTS_PATH}")


if __name__ == "__main__":
    main()
