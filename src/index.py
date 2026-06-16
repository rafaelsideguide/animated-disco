from array import array
from collections import Counter


class InvertedIndex:
    term_dict: dict[str, int]   # term -> term_id
    doc_lengths: list[int]      # internal_doc_id -> token count
    doc_ids: list[str]          # internal_doc_id -> original string id
    doc_meta: list[dict]        # internal_doc_id -> {"url": ..., "title": ...}
    avg_doc_len: float
    n_docs: int

    # Postings are stored CSR-style after finalize(): all term postings are
    # concatenated into two flat signed-int arrays, with post_offsets[term_id]
    # marking each term's slice. Three array objects total (instead of one list
    # of tuples per term) keeps both the pickle and memory small.
    post_doc_ids: array         # concatenated internal_doc_ids, grouped by term
    post_tfs: array             # parallel term frequencies
    post_offsets: array         # len n_terms+1; term_id -> [start, end) into the above

    def __init__(self):
        self.term_dict = {}
        self.doc_lengths = []
        self.doc_ids = []
        self.doc_meta = []
        self.avg_doc_len = 0.0
        self.n_docs = 0
        # Per-term builder, discarded by finalize() once flattened to CSR.
        self._postings: dict[int, tuple[array, array]] = {}
        self.post_doc_ids = array("i")
        self.post_tfs = array("i")
        self.post_offsets = array("i", [0])

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
                self._postings[term_id] = (array("i"), array("i"))
            else:
                term_id = self.term_dict[term]

            doc_ids, tfs = self._postings[term_id]
            doc_ids.append(internal_id)
            tfs.append(tf)

    def postings(self, term_id: int) -> tuple[array, array, int, int]:
        """Return (doc_ids, tfs, start, end) for a term: iterate indices
        [start, end) of the shared flat arrays. df == end - start."""
        start = self.post_offsets[term_id]
        end = self.post_offsets[term_id + 1]
        return self.post_doc_ids, self.post_tfs, start, end

    def finalize(self) -> None:
        self.n_docs = len(self.doc_ids)
        if self.n_docs > 0:
            self.avg_doc_len = sum(self.doc_lengths) / self.n_docs
        else:
            self.avg_doc_len = 0.0

        # Flatten the per-term builder into CSR arrays in term_id order.
        for term_id in range(len(self.term_dict)):
            doc_ids, tfs = self._postings[term_id]
            self.post_doc_ids.extend(doc_ids)
            self.post_tfs.extend(tfs)
            self.post_offsets.append(len(self.post_doc_ids))
        self._postings = {}

        # Downcast to 2-byte ints where the values fit (typical: <=65535 docs and
        # small term frequencies), roughly halving postings size. Falls back to
        # the wide 'i' arrays when they don't — readers are typecode-agnostic.
        if self.n_docs - 1 <= 0xFFFF:
            self.post_doc_ids = array("H", self.post_doc_ids)
        if not self.post_tfs or max(self.post_tfs) <= 0xFFFF:
            self.post_tfs = array("H", self.post_tfs)
