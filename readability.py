import os
import re
import random
import lxml.html
import numpy as np
from lxml.html.clean import Cleaner

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


basic_cleaner = Cleaner(forms=False, style=False, meta=False, page_structure=False, remove_unknown_tags=False, safe_attrs_only=False)

class L3SDocument(object):
    """A document from L3S dataset."""
    BLOCK_WRAPPER_REGEX = re.compile(r'<(table|dl|div|ol|ul|p|article|section)', re.I)
    BLANK_REGEX = re.compile(r'\s+')

    def __init__(self, original, annotated):
        self.original = original

        self.html_doc = HTMLLoader.from_file(original)
        self.main_content = self._get_main_content(annotated)
        print 'loading', self
        #print self.main_content

    def __repr__(self):
        return 'L3S Document <%s>' % (os.path.basename(self.original))

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
            else:
                children = span.getchildren()
                if len(children) == 1 and children[0].tag == 'span':
                    if 'x-text-density' in children[0].attrib and children[0].text:
                        text_pieces.append(children[0].text)

        return self.normalize_html_text(' '.join(text_pieces))

    def get_training_example(self, feature_extractor):
        """Get text block training example from this document.
        @return (feature_matrix, label_vector)
        """
        print 'generating traing examples', self

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

        if not article_blocks:
            print '[WARN]: nothing extracted from %s, returning all content' % self
            return self.normalize_html_text(self.html_doc.body.text_content())
        else:
            return '\n'.join(article_blocks)

    def get_main_content(self):
        return self.main_content

    TAGS_TO_IGNORE = set('style,script,option,object,embed,applet,link,noscript'.split(','))
    TAGS_INLINE = set('a,strike,u,b,i,em,strong,span,sup,code,tt,sub,var,abbr,acronym,font'.split(','))

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


class L3SEvaluator(object):
    """Evaluate content extraction based against trained model."""

    def __init__(self, docs, fe, model, scaler):
        """
        @param docs list<L3SDocument> to test
        @param fe feature extractor to use
        @param model classifier model trained
        @param scaler MatrixScaler to scale feature data
        """
        self.documents = docs
        self.feature_extractor = fe
        self.model = model
        self.scaler = scaler

    def evaluate(self):
        self.precisions = []
        self.recalls = []

        for doc in self.documents:
            features, _ = doc.get_training_example(self.feature_extractor)
            features = self.scaler.scale(np.array(features))
            classes = self.model.predict(features)

            extracted_content = doc.extract_article(classes)
            main_content = doc.get_main_content()

            # do token-level comparison
            e_words = set(extracted_content.split())
            m_words = set(main_content.split())
            common_words = e_words.intersection(m_words)
            if not common_words:
                common_words = set(['CANNOT_BELIEVE_THIS'])
                print 'WARN: no word predicted accurately for', doc

            self.precisions.append(1.0*len(common_words)/len(e_words))
            self.recalls.append(1.0*len(common_words)/len(m_words))

    @staticmethod
    def average(numbers):
        return 1.0 * sum(numbers) / len(numbers)

    def report(self, out_file=None):
        """
        @param out_file if set, will save data to the specified file
        """
        f_measures = [2/(1/p+1/r) for p,r in zip(self.precisions, self.recalls)]
        print 'Average precision', L3SEvaluator.average(self.precisions)
        print 'Average recall', L3SEvaluator.average(self.recalls)
        print 'Average F-measure', L3SEvaluator.average(f_measures)

        if out_file:
            with open(out_file, 'w') as pd:
                for f in f_measures:
                    pd.write('%.4f\n' % f)


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
            html = f.read().decode('utf-8', 'ignore')
            # lxml doesn't like xml encoding declaration in unicode html
            html = HTMLLoader.XML_ENCODING_DECLARATION.sub('', html)
            html = basic_cleaner.clean_html(html)
            return lxml.html.document_fromstring(html)
