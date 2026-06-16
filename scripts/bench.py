"""Benchmark + behavioral-oracle harness for perf work.

Captures search results (regression oracle), eval metrics, and timings so we can
prove optimizations don't change output. Usage:
    uv run python scripts/bench.py capture   # save oracle to /tmp/perf_oracle.json
    uv run python scripts/bench.py verify    # compare current results to oracle
"""
import sys, os
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time
import pickle
import pathlib

DATA = pathlib.Path(__file__).parent.parent / "data"
ORACLE = pathlib.Path("/tmp/perf_oracle.json")


def load_queries(path):
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


def run():
    t0 = time.perf_counter()
    with open(DATA / "index.pkl", "rb") as f:
        index = pickle.load(f)
    load_s = time.perf_counter() - t0

    from search import search
    import eval as eval_module

    queries = load_queries(DATA / "queries.jsonl")
    judgments = load_judgments(DATA / "judgments.jsonl")

    t1 = time.perf_counter()
    results = {
        qid: [doc_id for doc_id, _ in search(index, row["query"], k=100)]
        for qid, row in queries.items()
    }
    search_s = time.perf_counter() - t1

    metrics = eval_module.evaluate(results, judgments)
    size = (DATA / "index.pkl").stat().st_size
    return results, metrics, load_s, search_s, size


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    results, metrics, load_s, search_s, size = run()

    print(f"index load   : {load_s*1000:8.1f} ms")
    print(f"search (197q): {search_s*1000:8.1f} ms  ({search_s/len(results)*1000:.2f} ms/query)")
    print(f"index size   : {size:,} bytes")
    print(f"NDCG@10      : {metrics['ndcg@10']:.4f}  MRR={metrics['mrr']:.4f}  Recall@100={metrics['recall@100']:.4f}")

    if mode == "capture":
        ORACLE.write_text(json.dumps(results))
        print(f"\nOracle saved to {ORACLE} ({len(results)} queries)")
    elif mode == "verify":
        if not ORACLE.exists():
            print("\nNo oracle to verify against. Run `capture` first.")
            return
        golden = json.loads(ORACLE.read_text())
        diffs = [qid for qid in golden if golden[qid] != results.get(qid)]
        if diffs:
            print(f"\n!!! MISMATCH: {len(diffs)} queries differ from oracle: {diffs[:10]}")
            sys.exit(1)
        print(f"\nOK: all {len(golden)} query result lists IDENTICAL to oracle.")


if __name__ == "__main__":
    main()
