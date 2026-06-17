# Dense / kNN Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone dense (embedding) retriever — multilingual MiniLM embeddings indexed in hnswlib — that can be evaluated alongside BM25F.

**Architecture:** `src/dense.py` splits a model-free `VectorIndex` (hnswlib cosine) from a thin `Embedder` (sentence-transformers). `scripts/build_embeddings.py` embeds the corpus offline; `run_eval.py` gains `--retriever {bm25,dense}` with a lazy dense import so the default path stays torch-free. BM25F code is untouched.

**Tech Stack:** Python 3.11+, `sentence-transformers` (+torch), `hnswlib`, `numpy`, pytest, `uv`.

---

## File Structure

- **Create:** `src/dense.py` — `VectorIndex`, `Embedder`, `embed_text`, `dense_search`, `load_dense`.
- **Create:** `scripts/build_embeddings.py` — offline embedding build.
- **Create:** `tests/test_dense.py` — `VectorIndex`/`embed_text` unit tests + a skip-if-unavailable `Embedder` test.
- **Modify:** `scripts/run_eval.py` — `--retriever` flag, lazy dense import.
- **Modify:** `pyproject.toml` — `sentence-transformers`, `hnswlib` (already added via `uv add`).
- **Artifacts:** `data/hnsw.bin`, `data/dense_meta.pkl`.

Constants live in `src/dense.py`: `MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"`, `DIM = 384`, `BODY_EMBED_CHARS = 512`.

---

## Task 1: Confirm and commit dependencies

**Files:** Modify `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Ensure deps present (idempotent)**

Run:
```bash
uv add hnswlib sentence-transformers
```
Expected: both present under `[project] dependencies`; `torch` etc. resolved in `uv.lock`.

- [ ] **Step 2: Verify imports**

Run:
```bash
uv run python -c "import hnswlib, sentence_transformers; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add hnswlib + sentence-transformers for dense retrieval"
```

---

## Task 2: VectorIndex + embed_text (model-free core)

**Files:** Create `src/dense.py`, `tests/test_dense.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dense.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import tempfile
import numpy as np
from dense import VectorIndex, embed_text


class TestVectorIndex(unittest.TestCase):
    def _vecs(self):
        # doc "a" on axis0, "b" on axis1, "c" near "a".
        v = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
        return v, ["a", "b", "c"]

    def test_nearest_neighbor(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        res = idx.query(np.array([1, 0, 0, 0], dtype=np.float32), k=2)
        self.assertEqual([d for d, _ in res], ["a", "c"])

    def test_similarity_descending_and_unit_for_exact(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        res = idx.query(np.array([1, 0, 0, 0], dtype=np.float32), k=3)
        sims = [s for _, s in res]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertAlmostEqual(sims[0], 1.0, places=4)

    def test_k_truncates_and_caps(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        q = np.array([1, 0, 0, 0], dtype=np.float32)
        self.assertEqual(len(idx.query(q, k=2)), 2)
        self.assertEqual(len(idx.query(q, k=99)), 3)  # capped to corpus size

    def test_save_load_roundtrip(self):
        v, ids = self._vecs()
        idx = VectorIndex(dim=4)
        idx.build(v, ids)
        with tempfile.TemporaryDirectory() as d:
            idx.save(d)
            loaded = VectorIndex.load(d)
        res = loaded.query(np.array([0, 1, 0, 0], dtype=np.float32), k=1)
        self.assertEqual(res[0][0], "b")
        self.assertEqual(loaded.doc_ids, ids)

    def test_empty_index_returns_empty(self):
        idx = VectorIndex(dim=4)
        self.assertEqual(idx.query(np.array([1, 0, 0, 0], dtype=np.float32), k=5), [])


class TestEmbedText(unittest.TestCase):
    def test_concatenates_fields_and_caps_body(self):
        out = embed_text({"title": "T", "headings": "H", "body": "B" * 1000})
        self.assertTrue(out.startswith("T H "))
        self.assertLessEqual(len(out), 512 + 8)  # body capped at 512 chars

    def test_missing_fields(self):
        self.assertEqual(embed_text({}), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run python -m pytest tests/test_dense.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dense'`.

- [ ] **Step 3: Create `src/dense.py` with the core (no model yet)**

```python
import pickle
from pathlib import Path

import hnswlib
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384
BODY_EMBED_CHARS = 512

_EF_CONSTRUCTION = 200
_M = 16
_EF_QUERY = 64
_INDEX_FILE = "hnsw.bin"
_META_FILE = "dense_meta.pkl"


class VectorIndex:
    """hnswlib cosine index mapping integer labels -> doc_ids.

    Vectors need not be pre-normalized: the 'cosine' space normalizes internally,
    so query() returns cosine similarity (1 - cosine_distance)."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._index = None
        self.doc_ids: list[str] = []

    def build(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        n = len(doc_ids)
        assert vectors.shape == (n, self.dim), f"expected ({n}, {self.dim}), got {vectors.shape}"
        index = hnswlib.Index(space="cosine", dim=self.dim)
        index.init_index(max_elements=n, ef_construction=_EF_CONSTRUCTION, M=_M)
        index.add_items(vectors.astype(np.float32), np.arange(n))
        index.set_ef(_EF_QUERY)
        self._index = index
        self.doc_ids = list(doc_ids)

    def query(self, vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if self._index is None or not self.doc_ids:
            return []
        k = min(k, len(self.doc_ids))
        labels, distances = self._index.knn_query(vector.astype(np.float32), k=k)
        return [
            (self.doc_ids[int(lab)], 1.0 - float(dist))
            for lab, dist in zip(labels[0], distances[0])
        ]

    def save(self, directory) -> None:
        directory = Path(directory)
        self._index.save_index(str(directory / _INDEX_FILE))
        with open(directory / _META_FILE, "wb") as f:
            pickle.dump(
                {"doc_ids": self.doc_ids, "dim": self.dim, "model_name": MODEL_NAME}, f
            )

    @classmethod
    def load(cls, directory) -> "VectorIndex":
        directory = Path(directory)
        with open(directory / _META_FILE, "rb") as f:
            meta = pickle.load(f)
        obj = cls(dim=meta["dim"])
        obj.doc_ids = meta["doc_ids"]
        index = hnswlib.Index(space="cosine", dim=obj.dim)
        index.load_index(str(directory / _INDEX_FILE), max_elements=len(obj.doc_ids))
        index.set_ef(_EF_QUERY)
        obj._index = index
        return obj


def embed_text(fields: dict) -> str:
    """Build the per-document embedding input from parsed fields."""
    title = fields.get("title", "")
    headings = fields.get("headings", "")
    body = fields.get("body", "")[:BODY_EMBED_CHARS]
    return f"{title} {headings} {body}".strip()
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run python -m pytest tests/test_dense.py -v`
Expected: all PASS (`TestVectorIndex` + `TestEmbedText`).

- [ ] **Step 5: Commit**

```bash
git add src/dense.py tests/test_dense.py
git commit -m "Add VectorIndex (hnswlib cosine) and embed_text"
```

---

## Task 3: Embedder + dense_search + load_dense

**Files:** Modify `src/dense.py`, `tests/test_dense.py`

- [ ] **Step 1: Add the skip-if-unavailable Embedder test**

Append to `tests/test_dense.py`, before the `if __name__` block:

```python
class TestEmbedderIntegration(unittest.TestCase):
    def test_encode_shape_and_unit_norm(self):
        try:
            from dense import Embedder
            v = Embedder().encode(["hello world"])
        except Exception as e:  # model download/load unavailable (offline/CI)
            raise unittest.SkipTest(f"embedding model unavailable: {e}")
        self.assertEqual(v.shape, (1, 384))
        self.assertAlmostEqual(float(np.linalg.norm(v[0])), 1.0, places=3)
```

- [ ] **Step 2: Run it (passes if model cached, else skips)**

Run: `uv run python -m pytest tests/test_dense.py::TestEmbedderIntegration -v`
Expected: PASS if the model is available, otherwise SKIPPED — never FAIL.

- [ ] **Step 3: Append `Embedder`, `dense_search`, `load_dense` to `src/dense.py`**

```python
class Embedder:
    """Lazy sentence-transformers wrapper producing L2-normalized float32 vectors."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._ensure()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)


def dense_search(vindex: VectorIndex, embedder: Embedder, query: str, k: int = 10) -> list[tuple[str, float]]:
    vector = embedder.encode([query])[0]
    return vindex.query(vector, k=k)


def load_dense(directory) -> tuple[VectorIndex, Embedder]:
    return VectorIndex.load(directory), Embedder()
```

- [ ] **Step 4: Run the full dense suite**

Run: `uv run python -m pytest tests/test_dense.py -v`
Expected: all PASS (the integration test PASSES if model cached, else SKIPPED).

- [ ] **Step 5: Commit**

```bash
git add src/dense.py tests/test_dense.py
git commit -m "Add Embedder, dense_search, load_dense"
```

---

## Task 4: Offline embedding build script

**Files:** Create `scripts/build_embeddings.py`

- [ ] **Step 1: Create the script**

```python
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
```

- [ ] **Step 2: Smoke-check the script imports (do not run the full build yet)**

Run: `uv run python -c "import sys, os; sys.path.insert(0, 'scripts'); import build_embeddings; print('ok')"`
Expected: prints `ok` (no syntax/import errors). The full build runs in Task 6.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_embeddings.py
git commit -m "Add offline embedding build script"
```

---

## Task 5: Wire dense into run_eval

**Files:** Modify `scripts/run_eval.py`

- [ ] **Step 1: Add the `--retriever` flag and lazy dense branch**

In `scripts/run_eval.py`, replace this block at the top of `main()`:
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)

    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    results = {
        qid: [doc_id for doc_id, _score in search(index, row["query"], k=100)]
        for qid, row in queries.items()
    }
```
with:
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    args = parser.parse_args()

    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    if args.retriever == "dense":
        from dense import load_dense, dense_search
        vindex, embedder = load_dense(DATA)
        def retrieve(q):
            return dense_search(vindex, embedder, q, k=100)
    else:
        with open(DATA / "index.pkl", "rb") as f:
            index = pickle.load(f)
        def retrieve(q):
            return search(index, q, k=100)

    results = {
        qid: [doc_id for doc_id, _score in retrieve(row["query"])]
        for qid, row in queries.items()
    }
```
Leave the rest of `main()` (metrics, breakdown printing) unchanged.

- [ ] **Step 2: Verify the default (bm25) path is unchanged**

Run: `uv run python scripts/run_eval.py 2>&1 | tail -2`
Expected: same overall line as before — `NDCG@10=0.65  MRR=0.84  Recall@100=0.83  (n=197)`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_eval.py
git commit -m "run_eval: add --retriever {bm25,dense} with lazy dense import"
```

---

## Task 6: Build embeddings and evaluate dense

**Files:** Artifacts `data/hnsw.bin`, `data/dense_meta.pkl`

- [ ] **Step 1: Build embeddings**

Run: `uv run python scripts/build_embeddings.py`
Expected: streams the corpus, prints progress, ends with `Done. 50,000 docs, dim 384.` First run downloads the model (~470 MB) and may take several minutes (faster on Apple-Silicon MPS / CUDA).

- [ ] **Step 2: Evaluate the dense retriever**

Run: `uv run python scripts/run_eval.py --retriever dense --breakdown`
Expected: an overall + per-bucket table for dense retrieval. Record it.

- [ ] **Step 3: Compare to BM25F and sanity-check complementarity**

Reference BM25F (`--retriever bm25 --breakdown`): overall 0.65; paraphrase 0.44; short-keyword 0.70; code-identifier 0.61; non-english 0.82.

Success is **complementarity**, not beating BM25F solo (see spec). Confirm:
- the dense build + eval run end-to-end and produce sensible rankings;
- dense is relatively strong on **paraphrase** vs its own weaker buckets;
- dense and BM25F differ by bucket (the premise for hybrid / sub-project C).

Dense-alone overall NDCG below 0.65 is expected and is a pessimistic lower bound
(qrels are baseline-pooled + title/URL-graded; re-pooling is out of scope). Do
not treat a lower overall number as failure — report the breakdown and the
per-bucket comparison.

- [ ] **Step 4: Commit the artifacts**

```bash
git add data/hnsw.bin data/dense_meta.pkl
git commit -m "Build dense embeddings (multilingual MiniLM) + hnswlib index"
```

---

## Self-Review Notes

- **Spec coverage:** `VectorIndex` hnswlib cosine + save/load (Task 2); `embed_text` title+headings+body[:512] (Task 2); `Embedder` lazy/normalized (Task 3); `dense_search`/`load_dense` (Task 3); `build_embeddings.py` (Task 4); `run_eval --retriever` lazy import (Task 5); deps (Task 1); build+eval+complementarity check (Task 6). All spec sections covered.
- **Placeholder scan:** none — all code/commands concrete.
- **Type consistency:** `VectorIndex(dim)`, `.build(vectors, doc_ids)`, `.query(vector, k) -> [(doc_id, sim)]`, `.save(dir)`/`.load(dir)`; `Embedder.encode(texts) -> np.ndarray`; `embed_text(fields) -> str`; `dense_search(vindex, embedder, query, k)`; `load_dense(dir) -> (VectorIndex, Embedder)`. Constants `MODEL_NAME`/`DIM`/`BODY_EMBED_CHARS` defined once in `dense.py`. `run_eval` retriever closures return `[(doc_id, score)]` consistent with both `search` and `dense_search`.
```
