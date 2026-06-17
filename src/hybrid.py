# Rank-fusion utilities for combining ranked retrieval lists.


def rrf_fuse(ranked_lists, k=60, weights=None):
    """Reciprocal-rank fusion over already-sorted ranked lists.

    Score(d) = sum_i w_i / (k + rank_i(d)), rank_i 0-based within list i. A doc
    absent from a list contributes nothing from it. Input scores are IGNORED —
    only ranks matter, which makes RRF robust to incomparable score scales
    (BM25F magnitudes vs. cosine similarity). Returns (doc_id, score) sorted
    descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
        )

    scores = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def dedup_union(ranked_lists):
    """First-seen-order unique doc_ids across all ranked lists."""
    seen = set()
    union = []
    for ranked in ranked_lists:
        for doc_id, _score in ranked:
            if doc_id not in seen:
                seen.add(doc_id)
                union.append(doc_id)
    return union
