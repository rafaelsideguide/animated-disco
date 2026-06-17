import sys, os
# Strip scripts/ from sys.path so scripts/inspect.py doesn't shadow stdlib inspect.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
from pathlib import Path
from collections import defaultdict

from search import search
from dense import load_dense, dense_search
from rerank import load_doc_text, Reranker, rerank
from repool import new_candidates, grader_doc_text

DATA = Path(__file__).parent.parent / "data"
CANDIDATES_PATH = DATA / "repool_candidates.json"
POOL_DEPTH = 10


def load_queries():
    rows = {}
    with open(DATA / "queries.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def load_judged():
    judged = defaultdict(set)
    with open(DATA / "judgments.jsonl") as f:
        for line in f:
            r = json.loads(line)
            judged[r["qid"]].add(r["doc_id"])
    return judged


def main():
    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)
    vindex, embedder = load_dense(DATA)
    doc_text = load_doc_text(DATA)
    reranker = Reranker()
    reverse = {ext: i for i, ext in enumerate(index.doc_ids)}

    queries = load_queries()
    judged = load_judged()

    out = []
    total_new = 0
    empty_text = 0
    for qid, row in queries.items():
        q = row["query"]
        bm = [d for d, _ in search(index, q, k=100)]
        dn = [d for d, _ in dense_search(vindex, embedder, q, k=POOL_DEPTH)]
        # Rerank sees BM25F's full top-100 (not bm[:POOL_DEPTH]) so it can surface
        # rank-11..100 docs into its top-POOL_DEPTH — the point of reranking.
        rr = [d for d, _ in rerank(reranker, q, bm, doc_text, k=POOL_DEPTH)]
        new = new_candidates([bm[:POOL_DEPTH], dn, rr], judged[qid])
        if not new:
            continue
        docs = []
        for doc_id in new:
            i = reverse.get(doc_id)
            url = index.doc_meta[i].get("url", "") if i is not None else ""
            dt = doc_text.get(doc_id, "")
            if not dt:
                empty_text += 1
            docs.append({"doc_id": doc_id, "text": grader_doc_text(url, dt)})
        out.append({"qid": qid, "query": q, "docs": docs})
        total_new += len(docs)

    with open(CANDIDATES_PATH, "w") as f:
        json.dump(out, f)
    if empty_text:
        print(f"  WARNING: {empty_text} candidate docs had empty doc_text (corpus drift?)")
    print(f"Done. {len(out)} queries, {total_new:,} new candidate docs -> {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
