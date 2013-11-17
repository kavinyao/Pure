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

        # collect text blocks
        self._text_cache = []
        self._text_blocks = []
        self._traverse(self.html_doc.body)

        # generate features
        feature_matrix = feature_extractor.extract(self._text_blocks)
        # TODO: a text block must be longer than 3 words to be eligible for testing as main content
        labels = [1 if block.count(' ') > 2 and block in self.main_content else 0 for block in self._text_blocks]

        return feature_matrix, labels

    def extract_article(self, block_classes):
        article_blocks = []
        for i in xrange(len(block_classes)):
            if block_classes[i] == 1:
                article_blocks.append(self._text_blocks[i])

        return '\n'.join(article_blocks)

    TAGS_TO_IGNORE = set('style,script,option,object,embed,applet,link,noscript'.split(','))
    TAGS_INLINE = set('strike,u,b,i,em,strong,span,sup,code,tt,sub,var,abbr,acronym,font'.split(','))

    def _traverse(self, elem, depth=0):
        if elem.tag in L3SDocument.TAGS_TO_IGNORE:
            return

        flush = elem.tag not in L3SDocument.TAGS_INLINE
        if flush:
            self._generate_block()

        if elem.text:
            if elem.tag == 'a':
                self._text_cache.append(AnchorUtil.mark_as_anchor(elem.text))
            else:
                self._text_cache.append(elem.text)

        for child in elem.getchildren():
            self._traverse(child, depth+1)

        if flush:
            self._generate_block()

        if elem.tail:
            self._text_cache.append(elem.tail)

    def _generate_block(self):
        if not self._text_cache:
            return

        text_block = self.normalize_html_text(' '.join(self._text_cache))
        if text_block == '':
            return

        self._text_blocks.append(text_block)
        #print '%d. |%s|' % (self._labels[-1], text_block)

        self._text_cache = []

    def normalize_html_text(self, text):
        return L3SDocument.BLANK_REGEX.sub(' ', text).strip()


class AnchorUtil(object):
    BEGIN = '<<LINK<<'
    END = '>>LINK>>'
    extract_re = re.compile(BEGIN+'(.*?)'+END)
    ignore_re = re.compile(BEGIN+'|'+END)

    @staticmethod
    def mark_as_anchor(text):
        return AnchorUtil.BEGIN + text.strip() + AnchorUtil.END

    @staticmethod
    def extract_anchor_text(text):
        return ' '.join(AnchorUtil.extract_re.findall(text))

    @staticmethod
    def remove_markers(text):
        return AnchorUtil.ignore_re.sub('', text)


class DensitometricFeatureExtractor(object):
    LINE_DELIMITERS = '.,?!'

    @staticmethod
    def extract(text_blocks):
        """Extract number of words, text density and link density
        of the previous, current and next text blocks."""
        features = []
        for text_block in text_blocks:
            # remove link markers
            block = AnchorUtil.remove_markers(text_block)
            # crude word and lines count
            words = block.count(' ')
            lines = DensitometricFeatureExtractor.count_lines(block)
            sentences = DensitometricFeatureExtractor.count_sentences(block)

            # extract link text
            link_text = AnchorUtil.extract_anchor_text(text_block)
            link_words = link_text.count(' ')

            # prevent division by zero
            if words == 0: words = 1
            if lines == 0: lines = 1
            if sentences == 0: sentences = 1

            # number of words, average sentence length, text density, link density
            features.append([1.0*words, 1.0*words/sentences, 1.0*words/lines, 1.0*link_words/words])

        # add number of words quotient, avg. senence length quotient, text density quotient w.r.t. previous block
        n_features = len(features)
        for i in xrange(n_features):
            previous = features[i][:3] if i == 0 else features[i-1][:3]
            new_features = [features[i][j]/previous[j] for j in range(3)]
            features[i].extend(new_features)

        return features

    @staticmethod
    def count_sentences(text):
        sentences = 0
        for c in text:
            if c in DensitometricFeatureExtractor.LINE_DELIMITERS:
                sentences += 1

        return sentences

    @staticmethod
    def count_lines(text):
        return len(text) / 80


class HTMLLoader(object):
    XML_ENCODING_DECLARATION = re.compile(r'^\s*<\?xml[^>]*?>', re.I);

    @staticmethod
    def from_file(filename):
        with open(filename) as f:
            html = f.read().decode('utf-8')
            html = HTMLLoader.XML_ENCODING_DECLARATION.sub('', html)
            return lxml.html.document_fromstring(html)
