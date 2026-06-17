import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path

from repool import merge_grades

DATA = Path(__file__).parent.parent / "data"
JUDGMENTS_PATH = DATA / "judgments.jsonl"
GRADES_PATH = DATA / "repool_grades.jsonl"
SOURCE = "claude-code-pooled-2026"


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if not GRADES_PATH.exists():
        raise SystemExit(f"{GRADES_PATH} not found — run the grading step "
                         f"(build candidates, then grade them) before merging.")
    existing = _read_jsonl(JUDGMENTS_PATH) if JUDGMENTS_PATH.exists() else []
    grades = _read_jsonl(GRADES_PATH)
    new_rows = merge_grades(existing, grades, SOURCE)
    with open(JUDGMENTS_PATH, "a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Appended {len(new_rows):,} new judgments (source={SOURCE}); "
          f"judgments.jsonl now {len(existing) + len(new_rows):,} rows.")


if __name__ == "__main__":
    main()
