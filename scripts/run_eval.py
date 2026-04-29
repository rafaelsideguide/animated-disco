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

QUERY_TYPE_LABELS = {
    "natural":     "natural-language",
    "keyword":     "short-keyword",
    "paraphrase":  "paraphrase",
    "hyphenated":  "hyphenated",
    "code_id":     "code-identifier",
    "non_english": "non-english",
}


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


def print_metrics(label, m, n, width=18):
    print(
        f"{label:<{width}}  "
        f"NDCG@10={m['ndcg@10']:.2f}  "
        f"MRR={m['mrr']:.2f}  "
        f"Recall@100={m['recall@100']:.2f}  "
        f"(n={n})"
    )


def judgment_coverage(results, judgments, k=10):
    """Fraction of top-k results that have a judgment, averaged across queries."""
    fracs = []
    low = 0
    for qid, ranked in results.items():
        if qid not in judgments:
            continue
        top_k = ranked[:k]
        judged = sum(1 for d in top_k if d in judgments[qid])
        frac = judged / k if k else 0.0
        fracs.append(frac)
        if frac < 0.5:
            low += 1
    mean_frac = sum(fracs) / len(fracs) if fracs else 0.0
    return mean_frac, low, len(fracs)


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
    n_total = len([q for q in results if q in judgments])

    print(f"--- Overall ---")
    print_metrics("", metrics, n_total, width=0)

    if args.breakdown:
        # By intent
        print(f"\n--- By intent ---")
        groups = {}
        for qid, row in queries.items():
            if qid not in judgments:
                continue
            groups.setdefault(row.get("intent", "unknown"), []).append(qid)

        for intent in ["navigational", "informational", "transactional"]:
            qids = groups.get(intent, [])
            if not qids:
                continue
            m = eval_module.evaluate(
                {q: results[q] for q in qids},
                {q: judgments[q] for q in qids},
            )
            print_metrics(intent, m, len(qids))

        # By query type
        print(f"\n--- By query type ---")
        type_groups = {}
        for qid, row in queries.items():
            if qid not in judgments:
                continue
            type_groups.setdefault(row.get("query_type", "unknown"), []).append(qid)

        type_order = ["natural", "keyword", "paraphrase", "hyphenated", "code_id", "non_english"]
        for qt in type_order:
            qids = type_groups.get(qt, [])
            if not qids:
                continue
            m = eval_module.evaluate(
                {q: results[q] for q in qids},
                {q: judgments[q] for q in qids},
            )
            label = QUERY_TYPE_LABELS.get(qt, qt)
            print_metrics(label, m, len(qids))

        # Judgment coverage
        mean_frac, low, n_q = judgment_coverage(results, judgments)
        low_pct = 100 * low / n_q if n_q else 0
        print(f"\n--- Judgment coverage ---")
        print(f"Mean judged-fraction in top-10:      {mean_frac:.2f}")
        print(f"Queries with <50% judged in top-10:  {low} ({low_pct:.0f}%)")


if __name__ == "__main__":
    main()
