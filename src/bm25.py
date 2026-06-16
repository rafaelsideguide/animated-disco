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

    for token in query_tokens:
        if token not in index.term_dict:
            continue
        term_id = index.term_dict[token]
        doc_ids, post_tf, start, end = index.postings(term_id)
        df = end - start
        idf = log((n_docs - df + 0.5) / (df + 0.5) + 1)
        for i in range(start, end):
            doc = doc_ids[i]
            wtf = 0.0
            for f in FIELDS:
                tf = post_tf[f][i]
                if tf:
                    adl = avgdl[f]
                    ratio = field_lengths[f][doc] / adl if adl else 0.0
                    denom = 1 - B[f] + B[f] * ratio
                    wtf += BOOST[f] * tf / denom
            if wtf:
                scores[doc] = scores.get(doc, 0.0) + idf * wtf / (K1 + wtf)

    # nlargest == sorted(..., reverse=True)[:k] incl. stable tie-break.
    ranked = nlargest(k, scores.items(), key=lambda x: x[1])
    return [(index.doc_ids[doc], s) for doc, s in ranked]
