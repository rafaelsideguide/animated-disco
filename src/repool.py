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
    source. First grade for a (qid, doc_id) wins."""
    judged = {(r["qid"], r["doc_id"]) for r in existing_rows}
    seen = set()
    out = []
    for g in grade_rows:
        key = (g["qid"], g["doc_id"])
        if key in judged or key in seen:
            continue
        seen.add(key)
        out.append({"qid": g["qid"], "doc_id": g["doc_id"], "grade": g["grade"], "source": source})
    return out
