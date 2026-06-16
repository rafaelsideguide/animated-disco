import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from parse import clean_markdown, extract_headings, url_tokens, parse_document, BODY_CHAR_CAP


class TestCleanMarkdown(unittest.TestCase):
    def test_links_become_visible_text(self):
        self.assertEqual(clean_markdown("[hello](http://x.com) world"), "hello world")

    def test_images_removed(self):
        self.assertEqual(clean_markdown("![alt text](http://img.png) body"), "body")

    def test_bare_urls_removed(self):
        self.assertEqual(clean_markdown("visit https://example.com/x now"), "visit now")

    def test_headings_and_lists_flattened(self):
        self.assertEqual(clean_markdown("# Heading\n\n- one\n- two"), "Heading one two")

    def test_empty(self):
        self.assertEqual(clean_markdown(""), "")

    def test_preserves_underscores_in_identifiers(self):
        # snake_case identifiers must survive cleaning (tokenizer keeps underscores).
        self.assertEqual(clean_markdown("call get_user_by_id then guild_id"),
                         "call get_user_by_id then guild_id")

    def test_strips_stray_brackets(self):
        out = clean_markdown("text [a [b] c](url) end")
        self.assertNotIn("[", out)
        self.assertNotIn("]", out)


class TestExtractHeadings(unittest.TestCase):
    def test_collects_heading_text(self):
        self.assertEqual(extract_headings("# A\nbody text\n## B sub\n"), "A B sub")

    def test_no_headings(self):
        self.assertEqual(extract_headings("just body"), "")


class TestUrlTokens(unittest.TestCase):
    def test_host_and_path(self):
        self.assertEqual(
            url_tokens("https://www.revwatches.com/product/marine-star"),
            "revwatches product marine-star",
        )

    def test_empty(self):
        self.assertEqual(url_tokens(""), "")


class TestParseDocument(unittest.TestCase):
    def test_returns_all_fields(self):
        doc = {"title": "T", "url": "https://ex.com/a", "markdown": "# H\n[x](http://y) body"}
        fields = parse_document(doc)
        self.assertEqual(set(fields), {"title", "headings", "url", "body"})
        self.assertEqual(fields["title"], "T")
        self.assertEqual(fields["headings"], "H")
        self.assertEqual(fields["url"], "ex a")
        self.assertEqual(fields["body"], "H x body")

    def test_body_capped(self):
        doc = {"title": "", "url": "", "markdown": "word " * 1000}
        self.assertLessEqual(len(parse_document(doc)["body"]), BODY_CHAR_CAP)

    def test_missing_fields_default_empty(self):
        self.assertEqual(parse_document({}), {"title": "", "headings": "", "url": "", "body": ""})


if __name__ == "__main__":
    unittest.main()
