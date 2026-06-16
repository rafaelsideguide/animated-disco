import re
import unicodedata

import snowballstemmer

# Standard English stopword list (NLTK's list, minus the apostrophe-bearing
# contraction forms which can never match: the `\w+` tokenizer strips
# apostrophes, so "don't" tokenizes to "don" + "t" — both already listed).
# Embedded to avoid a runtime data download. Lowercase; matched post-normalization.
STOPWORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn",
})

_stemmer = snowballstemmer.stemmer("english")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    """NFKD-normalize, strip combining marks (accent-fold), and casefold."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return no_marks.casefold()


def _stem(token: str) -> str:
    """Stem alphabetic tokens longer than 2 chars; pass others through.

    The guard protects numbers, alphanumeric codes, and underscored
    identifiers (e.g. ``guild_id``, ``3c7wrnfl0ng288476``) from being mangled.
    """
    if len(token) > 2 and token.isalpha():
        return _stemmer.stemWord(token)
    return token


def tokenize(text: str) -> list[str]:
    normalized = _normalize(text)
    tokens = _TOKEN_RE.findall(normalized)
    return [_stem(t) for t in tokens if t not in STOPWORDS]
