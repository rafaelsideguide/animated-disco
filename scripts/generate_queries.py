#!/usr/bin/env python3
"""
Generate 200 search queries from the corpus for a Search/IR interview repo.

Output: data/queries.jsonl — lines of {"qid":"q001","query":"...","intent":"..."}

Usage:
    uv run python scripts/generate_queries.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

# Remove scripts/ from sys.path to prevent scripts/inspect.py shadowing stdlib inspect
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _scripts_dir]

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_PATH = Path("data/corpus.jsonl")
OUTPUT_PATH = Path("data/queries.jsonl")
BATCH_SIZE = 10  # docs per API call

QUERY_TYPES = [
    ("natural",     60),
    ("keyword",     40),
    ("paraphrase",  30),
    ("hyphenated",  30),
    ("code_id",     20),
    ("non_english", 20),
]

TECH_KEYWORDS = [
    "python", "javascript", "typescript", "java", "c++", "rust", "go",
    "docker", "kubernetes", "api", "function", "import", "class", "def ",
    "async", "await", "npm", "pip", "git", "sql", "http", "rest", "graphql",
    "react", "vue", "angular", "flask", "django", "fastapi", "lambda",
    "aws", "gcp", "azure", "terraform", "bash", "linux", "nginx", "postgres",
    "mongodb", "redis", "elasticsearch", "pytorch", "tensorflow", "llm",
    "embedding", "vector", "model", "train", "inference", "deploy",
]

TYPE_RULES = {
    "natural": (
        "Full natural language questions or how-to queries, 6-12 words. "
        "Example: 'how to deploy fastapi to aws lambda'"
    ),
    "keyword": (
        "Short 2-3 word keyword phrases only, no articles or filler words. "
        "Example: 'pytorch quantization'"
    ),
    "paraphrase": (
        "Rephrase the topic in a non-obvious way; avoid exact title words. "
        "Use synonyms or describe the problem instead. "
        "Example: 'making my model use less memory'"
    ),
    "hyphenated": (
        "Must include at least one hyphenated term or version number. "
        "Example: 'aws-lambda cold start', 'react-18 concurrent mode', 'multi-tenant saas'"
    ),
    "code_id": (
        "Must include a code identifier: camelCase name, snake_case name, function name, "
        "package name, or CLI flag. "
        "Example: 'useEffect dependency array', 'async def python', 'torch.nn.Module'"
    ),
    "non_english": (
        "Write the query in the same language as the document (not English). "
        "Match the document's natural language."
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus() -> list[dict]:
    docs = []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def lang_base(doc: dict) -> str:
    lang = doc.get("language", "") or ""
    if isinstance(lang, list):
        lang = lang[0] if lang else ""
    return lang.split("-")[0].lower()


def is_english(doc: dict) -> bool:
    return lang_base(doc) == "en"


def is_tech(doc: dict) -> bool:
    text = ((doc.get("markdown") or "") + " " + (doc.get("title") or "")).lower()
    return any(k in text for k in TECH_KEYWORDS)


def doc_snippet(doc: dict, max_chars: int = 300) -> str:
    md = (doc.get("markdown") or "").strip()
    return md[:max_chars].replace("\n", " ")


def build_prompt(query_type: str, docs: list[dict]) -> str:
    n = len(docs)
    rule = TYPE_RULES[query_type]

    doc_lines = []
    for doc in docs:
        doc_lines.append(
            f"- Title: {doc.get('title', '')}\n"
            f"  URL: {doc.get('url', '')}\n"
            f"  Snippet: {doc_snippet(doc)}"
        )
    docs_block = "\n".join(doc_lines)

    return (
        f"Given these web page summaries, generate {n} {query_type} search queries "
        f"a user might type to find this content.\n\n"
        f"Documents:\n{docs_block}\n\n"
        f"Rules for {query_type} queries:\n"
        f"- {rule}\n\n"
        f"Generate exactly {n} queries (one per document). "
        f"Output one query per line, no numbering, no explanation, no blank lines."
    )


def call_claude(client: anthropic.Anthropic, prompt: str) -> list[str]:
    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return lines
        except Exception as e:
            if attempt == 0:
                print(f"  API error: {e} — retrying in 5s…", file=sys.stderr)
                time.sleep(5)
            else:
                print(f"  API error on retry: {e} — skipping batch", file=sys.stderr)
                return []
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    print("Loading corpus…")
    all_docs = load_corpus()
    print(f"  {len(all_docs):,} documents loaded")

    english_tech = [d for d in all_docs if is_english(d) and is_tech(d)]
    non_english  = [d for d in all_docs if not is_english(d)]
    print(f"  English tech docs: {len(english_tech):,}")
    print(f"  Non-English docs:  {len(non_english):,}")

    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

    seen_queries: set[str] = set()
    results: list[dict] = []
    qid_counter = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Open output file for streaming writes
    out_file = open(OUTPUT_PATH, "w")

    def write_query(query: str, intent: str) -> bool:
        """Deduplicate and write; returns True if written."""
        nonlocal qid_counter
        q = query.strip()
        if not q or q.lower() in seen_queries:
            return False
        seen_queries.add(q.lower())
        qid_counter += 1
        qid = f"q{qid_counter:03d}"
        record = {"qid": qid, "query": q, "intent": intent}
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        results.append(record)
        return True

    for query_type, target_count in QUERY_TYPES:
        print(f"\n[{query_type}] generating {target_count} queries…")

        pool = non_english if query_type == "non_english" else english_tech
        if not pool:
            print(f"  No documents available for {query_type}, skipping.")
            continue

        generated = 0
        attempts = 0
        max_attempts = target_count * 3  # safety cap

        while generated < target_count and attempts < max_attempts:
            need = target_count - generated
            batch_size = min(BATCH_SIZE, need, len(pool))
            batch = random.sample(pool, batch_size)

            prompt = build_prompt(query_type, batch)
            lines = call_claude(client, prompt)

            for line in lines:
                if write_query(line, query_type):
                    generated += 1
                    if generated >= target_count:
                        break

            attempts += batch_size
            print(
                f"  {generated}/{target_count} generated "
                f"(API calls so far: {attempts // BATCH_SIZE + 1})",
                end="\r",
            )

            if generated < target_count:
                time.sleep(1)  # rate-limit courtesy pause

        print(f"  Done: {generated}/{target_count}")

    out_file.close()

    total = len(results)
    print(f"\nWrote {total} queries to {OUTPUT_PATH}")
    if total < 200:
        print(f"WARNING: only {total} queries generated (target was 200)")


if __name__ == "__main__":
    main()
