"""Spot-check: for a few queries, show BM25F top-5 vs reranked top-5 with the
judged grade and title, to distinguish 'reranker is bad' from 'reranker promotes
relevant-but-unjudged docs' (under-measurement)."""
import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
import pathlib
from collections import defaultdict

from search import search
from rerank import load_doc_text, Reranker, rerank

DATA = pathlib.Path(__file__).parent.parent / "data"


def main():
    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)
    doc_text = load_doc_text(DATA)
    reranker = Reranker()

    queries = {}
    with open(DATA / "queries.jsonl") as f:
        for line in f:
            r = json.loads(line)
            queries[r["qid"]] = r
    judg = defaultdict(dict)
    with open(DATA / "judgments.jsonl") as f:
        for line in f:
            r = json.loads(line)
            judg[r["qid"]][r["doc_id"]] = r["grade"]

    reverse = {ext: i for i, ext in enumerate(index.doc_ids)}

    def title(doc_id):
        i = reverse.get(doc_id)
        return index.doc_meta[i].get("title", "")[:70] if i is not None else "?"

    for qid in ["q101", "q131", "q002"]:  # a paraphrase, a hyphenated, an informational
        q = queries[qid]["query"]
        g = judg[qid]
        bm = [d for d, _ in search(index, q, k=100)]
        rr = [d for d, _ in rerank(reranker, q, bm, doc_text, k=100)]
        print(f"\n=== {qid} [{queries[qid]['query_type']}] {q!r}")
        print(f"  (judged docs for this query: {len(g)}; relevant: {sum(1 for v in g.values() if v>=1)})")
        print("  BM25F top-5:")
        for d in bm[:5]:
            print(f"    grade={g.get(d,'-'):>2}  {title(d)}")
        print("  Reranked top-5:")
        for d in rr[:5]:
            print(f"    grade={g.get(d,'-'):>2}  {title(d)}")


if __name__ == "__main__":
    main()
