# Cost-routing retriever: cheap BM25F first, escalate only when uncertain.
# TAU is the BM25F-confidence gate threshold; retune with scripts/tune_router.py.

from hybrid import rrf_fuse

TAU = 0.30


def bm25_margin(scored, k=10):
    """Normalized top-score margin (s0 - s_k) / s0 over the top-k BM25F scores.

    High margin = a clear top result = BM25F is confident. Returns 0.0 for
    empty/single-result/non-positive-top input (forces escalation)."""
    if not scored:
        return 0.0
    s0 = scored[0][1]
    if s0 <= 0 or len(scored) < 2:
        return 0.0
    idx = min(k, len(scored) - 1)
    sk = scored[idx][1]
    return (s0 - sk) / s0


def escalation_target(query):
    """Pick the escalation retriever: non-ASCII queries (≈ non-English) go to
    dense fusion (no cross-encoder); everything else to cross-encoder rerank."""
    if any(ord(c) > 127 for c in query):
        return "hybrid_rrf"
    return "bm25_rerank"


class CostRouter:
    """Routes each query to the cheapest sufficient retriever and tallies cost.

    Sub-retrievers are injected callables (so this is testable without models):
      bm25_fn(query)   -> [(id, score)]  BM25F top-100
      dense_fn(query)  -> [(id, score)]  dense top-100
      rerank_fn(query, candidate_ids) -> [(id, score)]  cross-encoder rerank
    """

    def __init__(self, bm25_fn, dense_fn, rerank_fn, tau=TAU):
        self.bm25_fn = bm25_fn
        self.dense_fn = dense_fn
        self.rerank_fn = rerank_fn
        self.tau = tau
        self.stats = {
            "bm25_only": 0,
            "hybrid_rrf": 0,
            "bm25_rerank": 0,
            "cross_encoder_calls": 0,
            "pairs_scored": 0,
        }

    def retrieve(self, query, k=100):
        scored = self.bm25_fn(query)
        if bm25_margin(scored) >= self.tau:
            self.stats["bm25_only"] += 1
            return scored[:k]
        if escalation_target(query) == "hybrid_rrf":
            dense = self.dense_fn(query)
            self.stats["hybrid_rrf"] += 1
            return rrf_fuse([scored, dense])[:k]
        candidates = [d for d, _ in scored]
        self.stats["bm25_rerank"] += 1
        self.stats["cross_encoder_calls"] += 1
        self.stats["pairs_scored"] += len(candidates)
        return self.rerank_fn(query, candidates)
