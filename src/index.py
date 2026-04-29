from collections import Counter


class InvertedIndex:
    term_dict: dict[str, int]                   # term -> term_id
    postings: dict[int, list[tuple[int, int]]]  # term_id -> [(internal_doc_id, tf)]
    doc_lengths: list[int]                      # internal_doc_id -> token count
    doc_ids: list[str]                          # internal_doc_id -> original string id
    doc_meta: list[dict]                        # internal_doc_id -> {"url": ..., "title": ...}
    avg_doc_len: float
    n_docs: int

    def __init__(self):
        self.term_dict = {}
        self.postings = {}
        self.doc_lengths = []
        self.doc_ids = []
        self.doc_meta = []
        self.avg_doc_len = 0.0
        self.n_docs = 0

    def add_document(self, doc_id: str, tokens: list[str], meta: dict) -> None:
        internal_id = len(self.doc_ids)

        self.doc_ids.append(doc_id)
        self.doc_meta.append(meta)
        self.doc_lengths.append(len(tokens))

        term_freqs = Counter(tokens)

        for term, tf in term_freqs.items():
            if term not in self.term_dict:
                term_id = len(self.term_dict)
                self.term_dict[term] = term_id
                self.postings[term_id] = []
            else:
                term_id = self.term_dict[term]

            self.postings[term_id].append((internal_id, tf))

    def finalize(self) -> None:
        self.n_docs = len(self.doc_ids)
        if self.n_docs > 0:
            self.avg_doc_len = sum(self.doc_lengths) / self.n_docs
        else:
            self.avg_doc_len = 0.0
