"""Helper for the agent-grading step of re-pooling.

  split  — pack repool_candidates.json into batch files of <= MAX_DOCS docs each
           (whole queries), under data/repool_batches/batch_<i>.json
  gather — concatenate data/repool_batches/grades_<i>.jsonl into
           data/repool_grades.jsonl and verify every candidate got one grade

Grading subagents read each batch_<i>.json and write grades_<i>.jsonl.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import glob
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
CANDIDATES_PATH = DATA / "repool_candidates.json"
BATCH_DIR = DATA / "repool_batches"
GRADES_PATH = DATA / "repool_grades.jsonl"
MAX_DOCS = 200


def _load_candidates():
    with open(CANDIDATES_PATH) as f:
        return json.load(f)


def split():
    BATCH_DIR.mkdir(exist_ok=True)
    # Clear prior batch AND grade files so a re-split never mixes stale grades.
    for old in glob.glob(str(BATCH_DIR / "batch_*.json")) + glob.glob(str(BATCH_DIR / "grades_*.jsonl")):
        os.remove(old)
    data = _load_candidates()
    batches, cur, cur_docs = [], [], 0
    for q in data:
        if cur and cur_docs + len(q["docs"]) > MAX_DOCS:
            batches.append(cur)
            cur, cur_docs = [], 0
        cur.append(q)
        cur_docs += len(q["docs"])
    if cur:
        batches.append(cur)
    for i, b in enumerate(batches):
        with open(BATCH_DIR / f"batch_{i}.json", "w") as f:
            json.dump(b, f)
    total = sum(len(q["docs"]) for q in data)
    print(f"Split {len(data)} queries / {total} docs into {len(batches)} batches "
          f"(<= {MAX_DOCS} docs each) in {BATCH_DIR}")


def gather():
    expected = {(q["qid"], d["doc_id"]) for q in _load_candidates() for d in q["docs"]}
    got, rows = set(), []
    for gf in sorted(glob.glob(str(BATCH_DIR / "grades_*.jsonl"))):
        with open(gf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r.get("qid"), r.get("doc_id"))
                if key in expected and key not in got and r.get("grade") in (0, 1, 2):
                    got.add(key)
                    rows.append({"qid": r["qid"], "doc_id": r["doc_id"], "grade": r["grade"]})
    with open(GRADES_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    missing = expected - got
    print(f"Wrote {len(rows)} grades -> {GRADES_PATH}")
    print(f"Coverage: {len(got)}/{len(expected)} candidates graded; {len(missing)} missing")
    if missing:
        miss_q = sorted({qid for qid, _ in missing})
        print(f"  MISSING in queries: {miss_q[:20]}{' ...' if len(miss_q) > 20 else ''}")
        sys.exit(1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    dispatch = {"split": split, "gather": gather}
    if cmd not in dispatch:
        sys.exit("usage: repool_batches.py [split|gather]")
    dispatch[cmd]()
