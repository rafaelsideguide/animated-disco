def new_candidates(pools: list[list[str]], judged: set) -> list[str]:
    """Union of ranked pools (first-seen order preserved), dropping any doc_id
    already in `judged` (a per-query set of already-judged doc_ids)."""
    seen = set()
    out = []
    for pool in pools:
        for doc_id in pool:
            if doc_id in judged or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
    return out


def grader_doc_text(url: str, doc_text: str) -> str:
    """Grader-facing text for a candidate: URL plus the cleaned doc text."""
    return f"URL: {url}\n{doc_text}".strip()


def merge_grades(existing_rows: list[dict], grade_rows: list[dict], source: str) -> list[dict]:
    """Return judgment rows to append: each (qid, doc_id) grade not already
    present in existing_rows (and not duplicated within grade_rows), tagged with
    source. First grade for a (qid, doc_id) wins. Rows missing qid/doc_id or whose
    grade is not 0/1/2 are skipped (defensive against partial grading output)."""
    judged = {(r["qid"], r["doc_id"]) for r in existing_rows}
    seen = set()
    out = []
    for g in grade_rows:
        qid, doc_id, grade = g.get("qid"), g.get("doc_id"), g.get("grade")
        if qid is None or doc_id is None or grade not in (0, 1, 2):
            continue
        key = (qid, doc_id)
        if key in judged or key in seen:
            continue
        seen.add(key)
        out.append({"qid": qid, "doc_id": doc_id, "grade": grade, "source": source})
    return out
