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
