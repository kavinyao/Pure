import re
from collections import Counter

CAPITAL_WORD_PATTERN = re.compile(r'([A-Z][a-z]+)')
DIGIT_CLASS = '<DIGITS>'
LONG_CLASS = '<LONG>'

def extract_css_tokens(attrib_text):
    """Extract tokens from id/class attribute text.
    E.g. "PageBody main-content post-123" => {"page", "body", "main", "content", "post", DIGIT_CLASS}
    @return Counter of tokens
    """
    classes = attrib_text.split()

    parts = classes
    for separator in '_-:':
        new_parts = []
        for part in parts:
            if separator in part:
                new_parts.extend(part.split(separator))
            else:
                new_parts.append(part)
        parts = new_parts

    tokens = []
    for part in parts:
        words = [w for w in CAPITAL_WORD_PATTERN.split(part) if w]
        if len(words) > 1:
            tokens.extend(word.lower() for word in words)
        else:
            tokens.append(part.lower())

    final_tokens = []
    for token in tokens:
        if not token: continue
        if token.isdigit():
            final_tokens.append(DIGIT_CLASS)
        elif len(token) > 16:
            final_tokens.append(LONG_CLASS)
        else:
            final_tokens.append(token)

    return Counter(final_tokens)


import unittest

class CSSTokenExtractionTest(unittest.TestCase):
    def test_empty(self):
        self.assertEquals(Counter(), extract_css_tokens('  '))

    def test_long(self):
        self.assertEquals(Counter({LONG_CLASS:1}), extract_css_tokens('3bae1be298abe9389b6e9789a'))

    def test_basic1(self):
        self.assertEqual(Counter(content=1), extract_css_tokens(' content '))

    def test_basic2(self):
        self.assertEqual(Counter(page=1, nav=1), extract_css_tokens('page-nav'))

    def test_basic3(self):
        self.assertEqual(Counter(page=1, nav=1), extract_css_tokens('PageNav'))

    def test_basic4(self):
        self.assertEqual(Counter(page=1, nav=1), extract_css_tokens('page_nav'))

    def test_basic5(self):
        self.assertEqual(Counter({DIGIT_CLASS:1}), extract_css_tokens('123'))

    def test_basic6(self):
        self.assertEqual(Counter(page=1, nav=1), extract_css_tokens('pageNav'))

    def test_basic7(self):
        self.assertEqual(Counter(page=1, nav=1), extract_css_tokens('page:nav'))

    def test_combination1(self):
        c = Counter(page=1, body=1, main=1, content=1, post=1)
        c.update({DIGIT_CLASS:1})
        self.assertEquals(c, extract_css_tokens('PageBody main-content post-123'))

    def test_combination2(self):
        c = Counter(page=2, body=1, main=1, content=1, post=1)
        c.update({DIGIT_CLASS:1})
        self.assertEquals(c, extract_css_tokens('pageBody_main-content page-post_123'))

if __name__ == '__main__':
    unittest.main()
