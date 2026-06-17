import pickle
from pathlib import Path

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_BODY_CHARS = 512
_DOC_TEXT_FILE = "doc_text.pkl"


class Reranker:
    """Lazy sentence-transformers CrossEncoder wrapper."""

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._ensure().predict([(query, t) for t in texts])
        return [float(s) for s in scores]


def rerank(reranker, query: str, candidate_ids: list[str], doc_text: dict, k: int = 10) -> list[tuple[str, float]]:
    """Reorder candidate_ids by cross-encoder score (descending), keep top-k.
    Missing doc_ids resolve to empty text (scored, not dropped)."""
    if not candidate_ids:
        return []
    texts = [doc_text.get(cid, "") for cid in candidate_ids]
    scores = reranker.score_pairs(query, texts)
    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:k]
    return [(cid, float(s)) for cid, s in ranked]


def load_doc_text(directory) -> dict:
    with open(Path(directory) / _DOC_TEXT_FILE, "rb") as f:
        return pickle.load(f)
