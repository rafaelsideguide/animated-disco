import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
from pathlib import Path

from tokenizer import tokenize
from index import InvertedIndex, FIELDS
from parse import parse_document

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus.jsonl"
INDEX_PATH = Path(__file__).parent.parent / "data" / "index.pkl"


def main():
    index = InvertedIndex()

    # Index documents in a single pass. The vocabulary is built incrementally by
    # add_document() (it populates term_dict), so a separate vocab-counting pass
    # would just re-read and re-tokenize the whole corpus for no benefit.
    print("Indexing documents...")
    total_docs = 0
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            fields = parse_document(doc)
            tokenized = {f: tokenize(fields[f]) for f in FIELDS}
            index.add_document(
                doc["id"],
                tokenized,
                {"url": doc["url"], "title": doc.get("title", "")},
            )
            total_docs += 1
            if total_docs % 10_000 == 0:
                print(f"  Indexed {total_docs:,} docs...")

    index.finalize()

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"\nDone.")
    print(f"  Vocab size : {len(index.term_dict):,}")
    print(f"  Total docs : {total_docs:,}")
    for f in FIELDS:
        print(f"  Avg {f} len: {index.avgdl[f]:.1f} tokens")


if __name__ == "__main__":
    main()
