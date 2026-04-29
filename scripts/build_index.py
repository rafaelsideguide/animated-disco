import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
from pathlib import Path

from tokenizer import tokenize
from index import InvertedIndex

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.jsonl"
INDEX_PATH = Path(__file__).parent.parent / "data" / "index.pkl"


def extract_text(doc: dict) -> str:
    title = doc.get("title") or ""
    body = doc.get("markdown") or ""
    return title + " " + body[:500]


def main():
    index = InvertedIndex()

    # Pass 1: build vocabulary
    print("Pass 1: building vocabulary...")
    vocab = set()
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            tokens = tokenize(extract_text(doc))
            vocab.update(tokens)
    print(f"  Vocabulary size: {len(vocab):,}")

    # Pass 2: index documents
    print("Pass 2: indexing documents...")
    total_docs = 0
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            tokens = tokenize(extract_text(doc))
            index.add_document(
                doc["id"],
                tokens,
                {"url": doc["url"], "title": doc.get("title", "")},
            )
            total_docs += 1
            if total_docs % 10_000 == 0:
                print(f"  Indexed {total_docs:,} docs...")

    index.finalize()

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    avg_len = sum(index.doc_lengths) / total_docs if total_docs else 0
    print(f"\nDone.")
    print(f"  Vocab size : {len(vocab):,}")
    print(f"  Total docs : {total_docs:,}")
    print(f"  Avg doc len: {avg_len:.1f} tokens")


if __name__ == "__main__":
    main()
