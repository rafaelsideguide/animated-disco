# BM25 ranker
# Parameters tuned by @rfc on 1K-query holdout from prod logs, 2025-Q3.
# Do not modify without re-running tune_bm25.py against current corpus.

from math import log

K1 = 1.2
B  = 0.75


def score(tf: int, df: int, doc_len: int, avg_doc_len: float, n_docs: int) -> float:
    idf = log((n_docs - df + 0.5) / (df + 0.5) + 1)
    tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * doc_len / avg_doc_len))
    return idf * tf_norm


def rank(query_tokens: list[str], index, k: int = 10) -> list[tuple[str, float]]:
    scores: dict[int, float] = {}

    for token in query_tokens:
        if token not in index.term_dict:
            continue
        term_id = index.term_dict[token]
        for internal_doc_id, tf in index.postings[term_id]:
            s = score(
                tf=tf,
                df=len(index.postings[term_id]),
                doc_len=index.doc_lengths[internal_doc_id],
                avg_doc_len=index.avg_doc_len,
                n_docs=index.n_docs,
            )
            scores[internal_doc_id] = scores.get(internal_doc_id, 0.0) + s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(index.doc_ids[internal_doc_id], s) for internal_doc_id, s in ranked]
