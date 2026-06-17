import re
from urllib.parse import urlparse

BODY_CHAR_CAP = 3000

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")     # ![alt](url)
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")     # [text](url) -> text
_URL_RE = re.compile(r"https?://\S+")               # bare urls
_LIST_MARKER_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.M)
_MD_NOISE_RE = re.compile(r"[#>*`|\[\]]")           # leftover markdown markers (keep underscores: snake_case)
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$", re.M)
_HEADING_LINE_FULL_RE = re.compile(r"^\s{0,3}#{1,6}.*$", re.M)
_WS_RE = re.compile(r"\s+")


def clean_markdown(md: str) -> str:
    """Strip markdown noise, keeping visible text. Order matters: images before
    links (image syntax contains link syntax), links before bare-URL removal."""
    if not md:
        return ""
    text = _IMAGE_RE.sub(" ", md)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _LIST_MARKER_RE.sub(" ", text)
    text = _MD_NOISE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def extract_headings(md: str) -> str:
    """Concatenated text of markdown heading lines (## ...), markdown-cleaned."""
    if not md:
        return ""
    cleaned = [clean_markdown(h) for h in _HEADING_LINE_RE.findall(md)]
    return " ".join(h for h in cleaned if h)


def url_tokens(url: str) -> str:
    """Host (minus leading www. and the TLD) plus the path, as space-joined words."""
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) > 1:
        host = " ".join(parts[:-1])
    text = host + " " + parsed.path.replace("/", " ")
    return _WS_RE.sub(" ", text.replace("_", " ")).strip()


def parse_document(doc: dict) -> dict:
    """Map a corpus doc to its four field strings. Heading lines are removed from
    the body so heading terms live only in the `headings` field (disjoint fields)."""
    md = doc.get("markdown") or ""
    body_md = _HEADING_LINE_FULL_RE.sub(" ", md)
    return {
        "title": doc.get("title") or "",
        "headings": extract_headings(md),
        "url": url_tokens(doc.get("url") or ""),
        "body": clean_markdown(body_md)[:BODY_CHAR_CAP],
    }
