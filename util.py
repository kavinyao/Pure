import re

CAPITAL_WORD_PATTERN = re.compile(r'([A-Z][a-z]+)')
DIGIT_CLASS = '<DIGITS>'

def extract_css_tokens(attrib_text):
    """Extract tokens from id/class attribute text.
    E.g. "PageBody main-content post-123" => {"page", "body", "main", "content", "post", DIGIT_CLASS}
    @return set of tokens
    """
    classes = attrib_text.split()

    parts = set()
    for cls in classes:
        if '_' in cls:
            parts.update(cls.split('_'))
        else:
            parts.add(cls)

    slug_parts = set()
    for part in parts:
        if '-' in part:
            slug_parts.update(part.split('-'))
        else:
            slug_parts.add(part)

    tokens = set()
    for part in slug_parts:
        words = [w for w in CAPITAL_WORD_PATTERN.split(part) if w]
        if len(words) > 1:
            tokens.update(word.lower() for word in words)
        else:
            tokens.add(part.lower())

    return set([DIGIT_CLASS if token.isdigit() else token for token in tokens])


import unittest

class CSSTokenExtractionTest(unittest.TestCase):
    def test_empty(self):
        self.assertEquals(set(), extract_css_tokens('  '))

    def test_basic1(self):
        self.assertEqual(set(['content']), extract_css_tokens(' content '))

    def test_basic2(self):
        self.assertEqual(set(['page', 'nav']), extract_css_tokens('page-nav'))

    def test_basic3(self):
        self.assertEqual(set(['page', 'nav']), extract_css_tokens('PageNav'))

    def test_basic4(self):
        self.assertEqual(set(['page', 'nav']), extract_css_tokens('page_nav'))

    def test_basic5(self):
        self.assertEqual(set([DIGIT_CLASS]), extract_css_tokens('123'))

    def test_basic6(self):
        self.assertEqual(set(['page', 'nav']), extract_css_tokens('pageNav'))

    def test_combination1(self):
        self.assertEquals(set(['page', 'body', 'main', 'content', 'post', DIGIT_CLASS]), extract_css_tokens('PageBody main-content post-123'))

    def test_combination2(self):
        self.assertEquals(set(['page', 'body', 'main', 'content', 'post', DIGIT_CLASS]), extract_css_tokens('pageBody_main-content page-post_123'))

if __name__ == '__main__':
    unittest.main()
