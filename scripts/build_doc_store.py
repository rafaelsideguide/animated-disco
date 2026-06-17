import sys, os
# Strip scripts/ from sys.path so scripts/inspect.py doesn't shadow stdlib inspect.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
from pathlib import Path

from parse import parse_document
from rerank import RERANK_BODY_CHARS, _DOC_TEXT_FILE

DATA = Path(__file__).parent.parent / "data"
CORPUS_PATH = DATA / "corpus.jsonl"
DOC_TEXT_PATH = DATA / _DOC_TEXT_FILE


def doc_store_text(doc: dict) -> str:
    fields = parse_document(doc)
    headings = fields["headings"][:RERANK_BODY_CHARS]
    body = fields["body"][:RERANK_BODY_CHARS]
    return f"{fields['title']} {headings} {body}".strip()


def main():
    doc_text = {}
    print("Building doc-text store...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            doc = json.loads(line)
            doc_text[doc["id"]] = doc_store_text(doc)
            if i % 10_000 == 0:
                print(f"  processed {i:,} docs...")
    with open(DOC_TEXT_PATH, "wb") as f:
        pickle.dump(doc_text, f)
    print(f"Done. {len(doc_text):,} docs -> {DOC_TEXT_PATH}")


if __name__ == "__main__":
    main()
