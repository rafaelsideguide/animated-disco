import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pickle
import json
import argparse
import pathlib

from search import search
import eval as eval_module

DATA = pathlib.Path(__file__).parent.parent / "data"


def load_jsonl(path):
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows[row["qid"]] = row
    return rows


def load_judgments(path):
    judgments = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["qid"]
            judgments.setdefault(qid, {})[row["doc_id"]] = row["grade"]
    return judgments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)

    queries = load_jsonl(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    results = {
        qid: [doc_id for doc_id, _score in search(index, row["query"], k=100)]
        for qid, row in queries.items()
    }

    metrics = eval_module.evaluate(results, judgments)
    print(f"NDCG@10    {metrics['ndcg@10']:.4f}")
    print(f"MRR        {metrics['mrr']:.4f}")
    print(f"Recall@100 {metrics['recall@100']:.4f}")

    if args.breakdown:
        by_intent = eval_module.evaluate_by_intent(results, judgments, queries)
        print("\n--- by intent ---")
        for intent, m in sorted(by_intent.items()):
            print(
                f"{intent:<20}"
                f"NDCG@10={m['ndcg@10']:.2f}  "
                f"MRR={m['mrr']:.2f}  "
                f"Recall@100={m['recall@100']:.2f}"
            )


if __name__ == "__main__":
    main()
