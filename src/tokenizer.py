STOPWORDS: set[str] = set()


def tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t not in STOPWORDS]
