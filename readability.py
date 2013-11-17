import os
import re
import lxml.html
import random

class L3SDocumentLoader(object):
    """Load L3S documents from specified directory."""

    def __init__(self, base_dir):
        """
        @param base_dir the root directory of L3S dataset
        """
        self.original_dir = os.path.join(base_dir, 'original')
        self.annotated_dir = os.path.join(base_dir, 'annotated')

    def get_documents(self, limit=0):
        files = os.listdir(self.original_dir)
        if limit > 0:
            files = random.sample(files, limit)

        documents = [self.generate_document(f) for f in files]
        return documents

    def generate_document(self, file_name):
        original = os.path.join(self.original_dir, file_name)
        annotated = os.path.join(self.annotated_dir, file_name)
        return L3SDocument(original, annotated)


class L3SDocument(object):
    """A document from L3S dataset."""
    BLOCK_WRAPPER_REGEX = re.compile(r'<(table|dl|div|ol|ul|p|article|section)', re.I)
    BLANK_REGEX = re.compile(r'\s+')

    def __init__(self, original, annotated):
        self.original = original

        self.html_doc = HTMLLoader.from_file(original)
        self.main_content = self._get_main_content(annotated)
        print 'processing', self.original
        #print self.main_content

    def _get_main_content(self, annotated):
        """Extract annotated main content.
        Note: currently cannot find a non-trivial way to handle
        the manually inserted tags, so just combine them.
        """
        doc = HTMLLoader.from_file(annotated)
        spans = doc.xpath(".//span[@class='x-nc-sel2']")
        text_pieces = []

        for span in spans:
            if span.text:
                text_pieces.append(span.text.strip())

        return self.normalize_html_text(' '.join(text_pieces))

    def get_training_example(self, feature_extractor):
        """Get text block training example from this document.
        @return (feature_matrix, label_vector)
        """
        print 'processing', self.original

        self._text_cache = []
        self._link_text_cache = []
        self._feature_matrix = []
        self._labels = []
        self._feature_extractor = feature_extractor
        self._traverse(self.html_doc.body)

        return self._feature_matrix, self._labels

    TAGS_TO_IGNORE = set('style,script,option,object,embed,applet,link,noscript'.split(','))
    TAGS_INLINE = set('strike,u,b,i,em,strong,span,sup,code,tt,sub,var,abbr,acronym,font'.split(','))

    def _traverse(self, elem, depth=0):
        if elem.tag in L3SDocument.TAGS_TO_IGNORE:
            return

        flush = elem.tag not in L3SDocument.TAGS_INLINE
        if flush:
            self._generate_feature()

        if elem.text:
            self._text_cache.append(elem.text)
            
            if elem.tag == 'a':
                self._link_text_cache.append(elem.text)

        for child in elem.getchildren():
            self._traverse(child, depth+1)

        if flush:
            self._generate_feature()

        if elem.tail:
            self._text_cache.append(elem.tail)

    def _generate_feature(self):
        if not self._text_cache:
            return

        text_block = self.normalize_html_text(' '.join(self._text_cache))
        link_block = self.normalize_html_text(' '.join(self._link_text_cache))
        if text_block == '':
            return

        self._feature_matrix.append(self._feature_extractor.extract(text_block, link_block))
        label = 1 if text_block in self.main_content else 0
        if text_block.count(' ') < 2:
            # too short to be confident
            label = 0
        self._labels.append(label)

        #print '%d. |%s|' % (self._labels[-1], text_block)

        self._text_cache = []
        self._link_text_cache = []

    def normalize_html_text(self, text):
        return L3SDocument.BLANK_REGEX.sub(' ', text).strip()


class DensitometricFeatureExtractor(object):
    LINE_DELIMITERS = '.,?!'

    @staticmethod
    def extract(text_block, link_block):
        """Extract number of words, text density and link density."""
        # crude word and lines count
        words = text_block.count(' ')
        lines = DensitometricFeatureExtractor.count_lines(text_block)

        link_words = link_block.count(' ')

        # prevent division by zero
        if words == 0: words = 1
        if lines == 0: lines = 1

        return [words, 1.0*words/lines, 1.0*link_words/words]

    @staticmethod
    def count_lines(text):
        lines = 0
        for c in text:
            if c in DensitometricFeatureExtractor.LINE_DELIMITERS:
                lines += 1

        return lines


class HTMLLoader(object):
    XML_ENCODING_DECLARATION = re.compile(r'^\s*<\?xml[^>]*?>', re.I);

    @staticmethod
    def from_file(filename):
        with open(filename) as f:
            html = f.read().decode('utf-8')
            html = HTMLLoader.XML_ENCODING_DECLARATION.sub('', html)
            return lxml.html.document_fromstring(html)
