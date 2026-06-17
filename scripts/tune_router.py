import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
import pathlib

import eval as eval_module
from search import search
from dense import load_dense, dense_search
from rerank import load_doc_text, Reranker, rerank
from hybrid import rrf_fuse
from route import bm25_margin, escalation_target

DATA = pathlib.Path(__file__).parent.parent / "data"
FLOOR = 0.74
TAUS = [i / 100 for i in range(0, 101, 5)]  # 0.00 .. 1.00 step 0.05


def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def load_judgments(path):
    j = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            j.setdefault(r["qid"], {})[r["doc_id"]] = r["grade"]
    return j


def main():
    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)
    vindex, embedder = load_dense(DATA)
    doc_text = load_doc_text(DATA)
    reranker = Reranker()

    # Precompute per query ONCE: BM25F ranking + margin, and the escalated
    # ranking (rerank or hybrid_rrf). Sweeping tau then only re-picks bm25 vs
    # escalated — no retriever re-runs.
    per_q = {}
    for qid, row in queries.items():
        q = row["query"]
        scored = search(index, q, k=100)
        target = escalation_target(q)
        if target == "hybrid_rrf":
            dense = dense_search(vindex, embedder, q, k=100)
            escalated = [d for d, _ in rrf_fuse([scored, dense])[:100]]
            uses_xenc = False
        else:
            candidates = [d for d, _ in scored]
            escalated = [d for d, _ in rerank(reranker, q, candidates, doc_text, k=100)]
            uses_xenc = True
        per_q[qid] = {
            "bm25": [d for d, _ in scored],
            "margin": bm25_margin(scored),
            "escalated": escalated,
            "uses_xenc": uses_xenc,
        }

    print(f"{'tau':>6} {'NDCG@10':>8} {'xenc_calls':>11} {'bm25_only':>10}")
    frontier = []
    for tau in TAUS:
        results, calls, bm25_only = {}, 0, 0
        for qid, d in per_q.items():
            if d["margin"] >= tau:
                results[qid] = d["bm25"]
                bm25_only += 1
            else:
                results[qid] = d["escalated"]
                if d["uses_xenc"]:
                    calls += 1
        ndcg = eval_module.evaluate(results, judgments)["ndcg@10"]
        frontier.append((tau, ndcg, calls))
        print(f"{tau:>6.2f} {ndcg:>8.3f} {calls:>11} {bm25_only:>10}")

    # Recommend: fewest cross-encoder calls subject to NDCG@10 >= FLOOR
    # (tie-break: lowest tau).
    eligible = [(calls, tau, ndcg) for tau, ndcg, calls in frontier if ndcg >= FLOOR]
    if eligible:
        calls, tau, ndcg = min(eligible)
        print(f"\nRecommended TAU={tau:.2f}  (NDCG@10={ndcg:.3f}, xenc_calls={calls}, floor={FLOOR})")
    else:
        print(f"\nNo tau meets the NDCG@10 floor of {FLOOR}; inspect the frontier above.")


if __name__ == "__main__":
    main()
