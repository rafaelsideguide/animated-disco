from array import array
from collections import Counter

FIELDS = ("title", "headings", "url", "body")


class InvertedIndex:
    term_dict: dict[str, int]   # term -> term_id
    doc_lengths: list[int]      # internal_doc_id -> total token count (display/compat)
    doc_ids: list[str]          # internal_doc_id -> original string id
    doc_meta: list[dict]
    field_lengths: dict         # field -> list[int] (internal_doc_id -> field token count)
    avgdl: dict                 # field -> average field length
    n_docs: int

    # CSR postings after finalize(): per term, the docs containing it in ANY field,
    # with parallel per-field term-frequency arrays. df == slice length.
    post_doc_ids: array
    post_tf: dict               # field -> array, parallel to post_doc_ids
    post_offsets: array         # term_id -> [start, end) into the above

    def __init__(self):
        self.term_dict = {}
        self.doc_lengths = []
        self.doc_ids = []
        self.doc_meta = []
        self.field_lengths = {f: [] for f in FIELDS}
        self.avgdl = {f: 0.0 for f in FIELDS}
        self.n_docs = 0
        # Builder: term_id -> {internal_doc_id -> [tf per field, FIELDS order]}.
        self._postings: dict[int, dict[int, list[int]]] = {}
        self.post_doc_ids = array("i")
        self.post_tf = {f: array("i") for f in FIELDS}
        self.post_offsets = array("i", [0])

    def add_document(self, doc_id: str, fields: dict, meta: dict) -> None:
        internal_id = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_meta.append(meta)

        total = 0
        for fi, fname in enumerate(FIELDS):
            tokens = fields.get(fname, [])
            self.field_lengths[fname].append(len(tokens))
            total += len(tokens)
            for term, tf in Counter(tokens).items():
                if term not in self.term_dict:
                    term_id = len(self.term_dict)
                    self.term_dict[term] = term_id
                    self._postings[term_id] = {}
                else:
                    term_id = self.term_dict[term]
                doc_map = self._postings[term_id]
                if internal_id not in doc_map:
                    doc_map[internal_id] = [0] * len(FIELDS)
                doc_map[internal_id][fi] = tf
        self.doc_lengths.append(total)

    def postings(self, term_id: int):
        """Return (doc_ids, post_tf, start, end): iterate indices [start, end) of
        the shared flat arrays; post_tf[field][i] is the tf in that field."""
        start = self.post_offsets[term_id]
        end = self.post_offsets[term_id + 1]
        return self.post_doc_ids, self.post_tf, start, end

    def finalize(self) -> None:
        self.n_docs = len(self.doc_ids)
        for f in FIELDS:
            lengths = self.field_lengths[f]
            self.avgdl[f] = (sum(lengths) / self.n_docs) if self.n_docs else 0.0

        for term_id in range(len(self.term_dict)):
            doc_map = self._postings[term_id]
            for internal_id in sorted(doc_map):
                self.post_doc_ids.append(internal_id)
                tfs = doc_map[internal_id]
                for fi, f in enumerate(FIELDS):
                    self.post_tf[f].append(tfs[fi])
            self.post_offsets.append(len(self.post_doc_ids))
        self._postings = {}

        # Downcast to 2-byte ints where values fit (readers are typecode-agnostic).
        if self.n_docs - 1 <= 0xFFFF:
            self.post_doc_ids = array("H", self.post_doc_ids)
        for f in FIELDS:
            arr = self.post_tf[f]
            if not arr or max(arr) <= 0xFFFF:
                self.post_tf[f] = array("H", arr)
