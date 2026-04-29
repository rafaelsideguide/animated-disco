from tokenizer import tokenize
from bm25 import rank


def search(index, query: str, k: int = 10) -> list[tuple[str, float]]:
    query_tokens = tokenize(query)
    return rank(query_tokens, index, k=k)


def get_doc(index, doc_id: str) -> dict | None:
    # Build a reverse lookup from string doc_id to internal integer id.
    # In a hot path you'd cache this; here we rebuild each call to stay simple.
    reverse = {ext_id: i for i, ext_id in enumerate(index.doc_ids)}

    internal_id = reverse.get(doc_id)
    if internal_id is None:
        return None

    result = dict(index.doc_meta[internal_id])
    result["id"] = doc_id
    result["doc_length"] = index.doc_lengths[internal_id]
    return result
