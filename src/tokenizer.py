import re
import unicodedata

import snowballstemmer

# Standard English stopword list (NLTK's 179-word list), embedded to avoid a
# runtime data download. Lowercase; matched after normalization.
STOPWORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
    "herself", "it", "it's", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "that'll", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn",
    "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",
    "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn",
    "wouldn't",
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
