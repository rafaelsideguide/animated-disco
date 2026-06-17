# BM25F ranker over fielded postings.
# Field boosts and per-field length-normalization (B) tuned by hand; retune with
# a train/test holdout (relevance option C) before trusting exact values.

from heapq import nlargest
from math import log

from index import FIELDS

K1 = 1.2   # single global saturation parameter (not per-field)
BOOST = {"title": 3.0, "url": 2.0, "headings": 1.5, "body": 1.0}
B = {"title": 0.0, "url": 0.2, "headings": 0.4, "body": 0.75}
assert set(BOOST) == set(FIELDS) and set(B) == set(FIELDS), "BOOST/B must cover exactly FIELDS"


def rank(query_tokens: list[str], index, k: int = 10) -> list[tuple[str, float]]:
    scores: dict[int, float] = {}

    n_docs = index.n_docs
    avgdl = index.avgdl
    field_lengths = index.field_lengths
    post_tf = index.post_tf

    # Per-field constants are identical for every term/posting in this query, so
    # bind them once to keep the inner loop free of dict lookups.
    field_params = [
        (BOOST[f], B[f], 1.0 - B[f], avgdl[f], field_lengths[f], post_tf[f])
        for f in FIELDS
    ]

    for token in query_tokens:
        if token not in index.term_dict:
            continue
        term_id = index.term_dict[token]
        doc_ids, _post_tf, start, end = index.postings(term_id)
        df = end - start
        idf = log((n_docs - df + 0.5) / (df + 0.5) + 1)
        for i in range(start, end):
            doc = doc_ids[i]
            wtf = 0.0
            for boost_f, b_f, one_minus_b, adl, flen, tf_arr in field_params:
                tf = tf_arr[i]
                if tf:
                    ratio = flen[doc] / adl if adl else 0.0
                    wtf += boost_f * tf / (one_minus_b + b_f * ratio)
            if wtf:
                scores[doc] = scores.get(doc, 0.0) + idf * (K1 + 1) * wtf / (K1 + wtf)

    ranked = nlargest(k, scores.items(), key=lambda x: x[1])
    return [(index.doc_ids[doc], s) for doc, s in ranked]
