# BM25 ranker
# Parameters tuned by @rfc on 1K-query holdout from prod logs, 2025-Q3.
# Do not modify without re-running tune_bm25.py against current corpus.

from heapq import nlargest
from math import log

K1 = 1.2
B  = 0.75


def score(tf: int, df: int, doc_len: int, avg_doc_len: float, n_docs: int) -> float:
    idf = log((n_docs - df + 0.5) / (df + 0.5) + 1)
    tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * doc_len / avg_doc_len))
    return idf * tf_norm


def rank(query_tokens: list[str], index, k: int = 10) -> list[tuple[str, float]]:
    scores: dict[int, float] = {}

    doc_lengths = index.doc_lengths
    avg_doc_len = index.avg_doc_len
    n_docs = index.n_docs

    for token in query_tokens:
        if token not in index.term_dict:
            continue
        term_id = index.term_dict[token]
        postings = index.postings[term_id]
        # idf depends only on the term, not the document — compute once per term.
        df = len(postings)
        idf = log((n_docs - df + 0.5) / (df + 0.5) + 1)
        for internal_doc_id, tf in postings:
            doc_len = doc_lengths[internal_doc_id]
            tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * doc_len / avg_doc_len))
            s = idf * tf_norm
            scores[internal_doc_id] = scores.get(internal_doc_id, 0.0) + s

    # nlargest is equivalent to sorted(..., reverse=True)[:k], including the
    # stable tie-break, so result ordering is identical to the previous sort.
    ranked = nlargest(k, scores.items(), key=lambda x: x[1])
    return [(index.doc_ids[internal_doc_id], s) for internal_doc_id, s in ranked]
