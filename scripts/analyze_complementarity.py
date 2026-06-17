"""One-off analysis: does dense surface judged-relevant docs BM25F misses?

For each query-type bucket, compares BM25F vs dense vs their union on recall of
JUDGED-relevant docs (grade>=1) at depth 10 and 100. union - bm25 = the recall
headroom a hybrid/rerank could capture. Operates only on judged docs, so it is
NOT subject to the unjudged-doc pool bias that depresses dense's NDCG.
"""
import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
import pathlib
from collections import defaultdict

from search import search
from dense import load_dense, dense_search

DATA = pathlib.Path(__file__).parent.parent / "data"


def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def load_judgments(path):
    j = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            j[r["qid"]][r["doc_id"]] = r["grade"]
    return j


def recall(topk_ids, relevant):
    if not relevant:
        return None
    return len(set(topk_ids) & relevant) / len(relevant)


def main():
    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)
    vindex, embedder = load_dense(DATA)

    # qid -> ranked doc lists
    bm25 = {q: [d for d, _ in search(index, r["query"], k=100)] for q, r in queries.items()}
    dense = {q: [d for d, _ in dense_search(vindex, embedder, r["query"], k=100)] for q, r in queries.items()}

    buckets = defaultdict(list)
    for qid, r in queries.items():
        if qid in judgments:
            buckets[r.get("query_type", "unknown")].append(qid)

    order = ["natural", "keyword", "paraphrase", "hyphenated", "code_id", "non_english"]
    print(f"{'bucket':<14} {'n':>3}  {'B@100':>6} {'D@100':>6} {'U@100':>6} {'+dense':>7}   {'B@10':>5} {'U@10':>5} {'+dense':>7}")
    for bucket in order + [b for b in buckets if b not in order]:
        qids = buckets.get(bucket, [])
        if not qids:
            continue
        rows = []
        for qid in qids:
            rel = {d for d, g in judgments[qid].items() if g >= 1}
            if not rel:
                continue
            b100, d100 = set(bm25[qid][:100]), set(dense[qid][:100])
            b10, d10 = set(bm25[qid][:10]), set(dense[qid][:10])
            rows.append((
                recall(b100, rel), recall(d100, rel), recall(b100 | d100, rel),
                recall(b10, rel), recall(b10 | d10, rel),
            ))
        n = len(rows)
        b100 = sum(r[0] for r in rows) / n
        d100 = sum(r[1] for r in rows) / n
        u100 = sum(r[2] for r in rows) / n
        b10 = sum(r[3] for r in rows) / n
        u10 = sum(r[4] for r in rows) / n
        print(f"{bucket:<14} {n:>3}  {b100:>6.2f} {d100:>6.2f} {u100:>6.2f} {u100-b100:>+7.2f}   {b10:>5.2f} {u10:>5.2f} {u10-b10:>+7.2f}")


if __name__ == "__main__":
    main()
