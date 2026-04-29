import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pickle
import argparse
from pathlib import Path

from search import search, get_doc

INDEX_PATH = Path(__file__).parent.parent / "data" / "index.pkl"
QUERIES_PATH = Path(__file__).parent.parent / "data" / "queries.jsonl"
JUDGMENTS_PATH = Path(__file__).parent.parent / "data" / "judgments.jsonl"


def load_index():
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def load_queries():
    queries = {}
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["qid"]] = q
    return queries


def load_judgments():
    judgments = {}
    with open(JUDGMENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            qid = j["qid"]
            if qid not in judgments:
                judgments[qid] = {}
            judgments[qid][j["doc_id"]] = j["grade"]
    return judgments


def cmd_term(index, args):
    term = args.term
    if term not in index.term_dict:
        print(f"Term '{term}' not found in index.")
        return
    term_id = index.term_dict[term]
    postings = index.postings[term_id]
    df = len(postings)
    print(f"Term: '{term}'")
    print(f"  df (docs containing term): {df:,}")
    top5 = sorted(postings, key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 docs by tf:")
    for internal_id, tf in top5:
        meta = index.doc_meta[internal_id]
        doc_id = index.doc_ids[internal_id]
        url = meta.get("url", "")
        print(f"    doc_id={doc_id}  tf={tf}  url={url}")


def get_meta(index, doc_id_to_internal, doc_id):
    internal = doc_id_to_internal.get(doc_id)
    if internal is None:
        return {}
    return index.doc_meta[internal]


def get_length(index, doc_id_to_internal, doc_id):
    internal = doc_id_to_internal.get(doc_id)
    if internal is None:
        return 0
    return index.doc_lengths[internal]


def cmd_doc(index, doc_id_to_internal, args):
    doc_id = args.doc_id
    doc = get_doc(index, doc_id)
    if doc is None:
        print(f"Doc '{doc_id}' not found in index.")
        return
    meta = get_meta(index, doc_id_to_internal, doc_id)
    length = get_length(index, doc_id_to_internal, doc_id)
    print(f"Doc: {doc_id}")
    print(f"  url        : {meta.get('url', '')}")
    print(f"  title      : {meta.get('title', '')}")
    print(f"  doc_length : {length}")


def cmd_query(index, doc_id_to_internal, args):
    queries = load_queries()
    judgments = load_judgments()
    qid = args.qid
    if qid not in queries:
        print(f"Query '{qid}' not found.")
        return
    query_text = queries[qid]["query"]
    print(f"Query [{qid}]: {query_text}")
    results = search(index, query_text, k=10)
    qjudgments = judgments.get(qid, {})
    print(f"{'Rank':<5} {'Score':<10} {'Grade':<7} {'Doc ID':<20} URL")
    for rank, (doc_id, score) in enumerate(results, 1):
        grade = qjudgments.get(doc_id, "-")
        meta = get_meta(index, doc_id_to_internal, doc_id)
        url = meta.get("url", "")
        print(f"{rank:<5} {score:<10.4f} {str(grade):<7} {doc_id:<20} {url}")


def cmd_judgments(index, doc_id_to_internal, args):
    judgments = load_judgments()
    qid = args.qid
    if qid not in judgments:
        print(f"No judgments found for query '{qid}'.")
        return
    qjudgments = judgments[qid]
    sorted_docs = sorted(qjudgments.items(), key=lambda x: x[1], reverse=True)
    print(f"Judgments for query '{qid}':")
    print(f"{'Grade':<7} {'Doc ID':<20} URL")
    for doc_id, grade in sorted_docs:
        meta = get_meta(index, doc_id_to_internal, doc_id)
        url = meta.get("url", "")
        print(f"{grade:<7} {doc_id:<20} {url}")


def main():
    parser = argparse.ArgumentParser(description="Inspect the search index and results.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_term = subparsers.add_parser("term", help="Look up a term in the index.")
    p_term.add_argument("term", type=str)

    p_doc = subparsers.add_parser("doc", help="Look up a document by ID.")
    p_doc.add_argument("doc_id", type=str)

    p_query = subparsers.add_parser("query", help="Run a query and show ranked results.")
    p_query.add_argument("qid", type=str)

    p_judgments = subparsers.add_parser("judgments", help="Show relevance judgments for a query.")
    p_judgments.add_argument("qid", type=str)

    args = parser.parse_args()
    index = load_index()
    doc_id_to_internal = {doc_id: i for i, doc_id in enumerate(index.doc_ids)}

    if args.command == "term":
        cmd_term(index, args)
    elif args.command == "doc":
        cmd_doc(index, doc_id_to_internal, args)
    elif args.command == "query":
        cmd_query(index, doc_id_to_internal, args)
    elif args.command == "judgments":
        cmd_judgments(index, doc_id_to_internal, args)


if __name__ == "__main__":
    main()
