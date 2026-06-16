import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from tokenizer import tokenize, STOPWORDS


class TestTokenize(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(tokenize("Python WEB"), ["python", "web"])

    def test_splits_hyphens(self):
        # Hyphenated query terms must split so they match space-separated corpus text.
        self.assertEqual(tokenize("treasury-yields"), ["treasuri", "yield"])

    def test_splits_en_and_em_dashes(self):
        self.assertEqual(tokenize("alpha–beta—gamma"), ["alpha", "beta", "gamma"])

    def test_strips_punctuation(self):
        self.assertEqual(tokenize("hello, world!"), ["hello", "world"])

    def test_splits_on_dots(self):
        self.assertEqual(tokenize("node.js"), ["node", "js"])

    def test_accent_folding(self):
        self.assertEqual(tokenize("café"), ["cafe"])

    def test_removes_stopwords(self):
        # "the", "of", "a" are stopwords; "manage"/"containers" survive (stemmed).
        self.assertEqual(tokenize("how to manage the containers"), ["manag", "contain"])

    def test_stems_word_forms_to_common_root(self):
        # Different surface forms collapse to the same stem (paraphrase matching).
        self.assertEqual(tokenize("managing"), tokenize("manages"))
        self.assertEqual(tokenize("managing"), tokenize("management"))

    def test_preserves_underscored_identifiers(self):
        # code_id queries rely on underscores staying joined.
        self.assertEqual(tokenize("guild_id financial_report"), ["guild_id", "financial_report"])

    def test_preserves_numbers_and_codes(self):
        # Numbers and alphanumeric codes must not be stemmed or dropped.
        self.assertEqual(tokenize("april-2025"), ["april", "2025"])
        self.assertEqual(tokenize("3c7wrnfl0ng288476"), ["3c7wrnfl0ng288476"])

    def test_short_alpha_tokens_not_stemmed(self):
        # Guard: tokens <= 2 chars or non-alpha are passed through unstemmed.
        self.assertEqual(tokenize("go"), ["go"])

    def test_stopwords_filtered_before_stemming(self):
        # "only" is a stopword, but its stem "onli" is not. Filtering must run on
        # the raw token (pre-stem), so "only" is dropped entirely. Pins the order.
        self.assertEqual(tokenize("only"), [])

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_whitespace_only(self):
        self.assertEqual(tokenize("   \t\n "), [])

    def test_all_stopwords(self):
        self.assertEqual(tokenize("the of a an"), [])

    def test_stopwords_is_nonempty(self):
        self.assertIn("the", STOPWORDS)
        self.assertGreater(len(STOPWORDS), 100)


if __name__ == "__main__":
    unittest.main()
