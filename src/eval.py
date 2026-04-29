from math import log2


def ndcg_at_k(ranked_ids: list[str], grades: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        grades.get(ranked_ids[i], 0) / log2(i + 2)
        for i in range(min(k, len(ranked_ids)))
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr(ranked_ids: list[str], grades: dict[str, int]) -> float:
    for rank, doc_id in enumerate(ranked_ids):
        if grades.get(doc_id, 0) >= 1:
            return 1.0 / (rank + 1)
    return 0.0


def recall_at_k(ranked_ids: list[str], grades: dict[str, int], k: int = 100) -> float:
    relevant = sum(1 for g in grades.values() if g >= 1)
    if relevant == 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if grades.get(doc_id, 0) >= 1)
    return hits / relevant


def evaluate(results: dict[str, list[str]], judgments: dict[str, dict[str, int]]) -> dict:
    """Average NDCG@10, MRR, and Recall@100 across all qids that have judgments."""
    qids = [qid for qid in results if qid in judgments]
    if not qids:
        return {"ndcg@10": 0.0, "mrr": 0.0, "recall@100": 0.0}

    n = len(qids)
    return {
        "ndcg@10":    sum(ndcg_at_k(results[q], judgments[q])    for q in qids) / n,
        "mrr":        sum(mrr(results[q], judgments[q])           for q in qids) / n,
        "recall@100": sum(recall_at_k(results[q], judgments[q])  for q in qids) / n,
    }


def evaluate_by_intent(
    results: dict[str, list[str]],
    judgments: dict[str, dict[str, int]],
    queries: dict[str, dict],
) -> dict[str, dict]:
    """Same as evaluate() but broken down by the intent field in queries[qid]."""
    groups: dict[str, list[str]] = {}
    for qid in results:
        if qid not in judgments:
            continue
        intent = queries.get(qid, {}).get("intent", "unknown")
        groups.setdefault(intent, []).append(qid)

    return {
        intent: evaluate(
            {q: results[q] for q in qids},
            {q: judgments[q] for q in qids},
        )
        for intent, qids in groups.items()
    }
