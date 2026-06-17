import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path

import numpy as np

from parse import parse_document
from dense import VectorIndex, Embedder, embed_text

DATA = Path(__file__).parent.parent / "data"
CORPUS_PATH = DATA / "corpus.jsonl"
BATCH = 256


def main():
    embedder = Embedder()
    doc_ids: list[str] = []
    texts: list[str] = []
    chunks: list[np.ndarray] = []

    def flush():
        if texts:
            chunks.append(embedder.encode(texts))
            texts.clear()

    print("Embedding documents...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            doc = json.loads(line)
            doc_ids.append(doc["id"])
            texts.append(embed_text(parse_document(doc)))
            if len(texts) >= BATCH:
                flush()
            if i % 10_000 == 0:
                print(f"  embedded {i:,} docs...")
    flush()

    vectors = np.vstack(chunks)
    print(f"  vectors: {vectors.shape}")

    print("Building hnsw index...")
    vindex = VectorIndex()
    vindex.build(vectors, doc_ids)
    vindex.save(DATA)
    print(f"Done. {len(doc_ids):,} docs, dim {vindex.dim}.")


if __name__ == "__main__":
    main()
