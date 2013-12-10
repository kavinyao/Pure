import os
import re
import random
import lxml.html
import numpy as np
from sklearn import svm
from lxml.html.clean import Cleaner

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

class DocumentLoader(object):
    """Load documents from specified directory."""

    def __init__(self, base_dir, original_subdir, annotated_subdir):
        """
        @param base_dir the root directory of dataset
        """
        self.original_dir = os.path.join(base_dir, original_subdir)
        self.annotated_dir = os.path.join(base_dir, annotated_subdir)

    def get_documents(self, limit=0):
        files = os.listdir(self.original_dir)
        if limit > 0:
            files = random.sample(files, limit)

        documents = [self.generate_document(f) for f in files]
        return documents

    def generate_document(self, file_name):
        original = os.path.join(self.original_dir, file_name)
        annotated = os.path.join(self.annotated_dir, file_name)

        return self._new_document(original, annotated)

    def _new_document(self, original, annotated):
        raise NotImplementedError


class L3SDocumentLoader(DocumentLoader):
    def __init__(self, base_dir):
        super(L3SDocumentLoader, self).__init__(base_dir, 'original', 'annotated')

    def _new_document(self, original, annotated):
        return L3SDocument(original, annotated)


class DragnetDocumentLoader(DocumentLoader):
    def __init__(self, base_dir):
        super(DragnetDocumentLoader, self).__init__(base_dir, 'HTML', 'Corrected')

    def _new_document(self, original, annotated):
        return DragnetDocument(original, '%s.corrected.txt' % annotated)


class Block(object):
    def __init__(self, label, text, tokens):
        """
        @param label POSITIVE_LABEL or NEGATIVE_LABEL
        @param text text with possibly inserted anchor markers
        @param tokens words extracted from id/class attribute
        """
        self.label = label
        self.text = text
        self.tokens = tokens


class Document(object):
    """A generic HTML document for extraction."""
    BLANK_REGEX = re.compile(r'\s+')

    def __init__(self, original, annotated):
        """
        @param original path to original HTML
        @param annotated path to gold corresponding gold standard
        """
        self.original = original

        print 'loading', self
        self.html_doc = HTMLLoader.from_file(original)
        self.main_content = self._get_main_content(annotated)
        #print self.main_content

        # for collecting text blocks
        self._text_cache = []
        self._text_blocks = []
        self._text_blocks_generated = False

    def _get_main_content(self, annotated):
        raise NotImplementedError

    def __repr__(self):
        return 'Document <%s>' % (os.path.basename(self.original))

    def get_blocks(self):
        """Get text blocks from this document.
        @return list<Block>
        """
        if self._text_blocks_generated:
            return self._text_blocks

        print 'generating text blocks', self
        self._traverse(self.html_doc.body)
        self._text_blocks_generated = True

        return self._text_blocks

    def extract_article(self, block_classes):
        article_blocks = []
        for i in xrange(len(block_classes)):
            if block_classes[i] == POSITIVE_LABEL:
                article_blocks.append(AnchorUtil.remove_markers(self._text_blocks[i].text))

        if not article_blocks:
            print '[WARN]: nothing extracted from %s, returning all content' % self
            return self.normalize_html_text(self.html_doc.body.text_content())
        else:
            return '\n'.join(article_blocks)

    def get_main_content(self):
        return self.main_content

    TAGS_TO_IGNORE = set('style,script,option,object,embed,applet,link,noscript'.split(','))
    TAGS_INLINE = set('a,b,big,i,small,tt,abbr,acronym,cite,code,dfn,em,kbd,strong,samp,var,bdo,br,img,map,object,q,span,sub,sup,button,input,label,select,textarea'.split(','))

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

        text = self.normalize_html_text(' '.join(self._text_cache))
        if text == '':
            self._text_cache = []
            return

        # TODO: a text block must be longer than 3 words to be eligible for testing as main content
        label = POSITIVE_LABEL if text.count(' ') > 2 and AnchorUtil.remove_markers(text) in self.main_content else NEGATIVE_LABEL
        self._text_blocks.append(Block(label, text, None))

        self._text_cache = []

    def normalize_html_text(self, text):
        return L3SDocument.BLANK_REGEX.sub(' ', text).strip()


class L3SDocument(Document):
    """A document from L3S dataset."""

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


def read_unicode_from(file):
    with open(file) as f:
        content = f.read()
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return content.decode('utf-16')
            except UnicodeDecodeError:
                try:
                    return content.decode('windows-1252')
                except UnicodeDecodeError:
                    try:
                        return content.decode('iso-8859-1')
                    except UnicodeDecodeError:
                        print 'Warning: cannot detect encoding of %f, using utf-8 ignore mode...'
                        return content.decode('utf-8', 'ignore')


class DragnetDocument(Document):
    """A document from Dragnet 2012 dataset."""

    def __repr__(self):
        return 'Dragnet Document <%s>' % (os.path.basename(self.original))

    def _get_main_content(self, annotated):
        """Extract annotated main content.
        """
        return self.normalize_html_text(read_unicode_from(annotated))


class Evaluator(object):
    """Evaluate content extraction based against trained model."""

    def __init__(self, docs, model):
        """
        @param docs list<Document> to test
        @param model an extraction model with predict method
        """
        self.documents = docs
        self.model = model

    def evaluate(self):
        self.precisions = []
        self.recalls = []

        for doc in self.documents:
            classes = self.model.predict(doc)
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
        print 'Average precision', Evaluator.average(self.precisions)
        print 'Average recall', Evaluator.average(self.recalls)
        print 'Average F-measure', Evaluator.average(f_measures)

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
    n_features = 7

    @staticmethod
    def extract(text_blocks):
        """Extract number of words, text density and link density
        of the previous, current and next text blocks."""
        features = []
        for text_block in text_blocks:
            # remove link markers
            block = AnchorUtil.remove_markers(text_block.text)
            # crude word and lines count
            words = block.count(' ')
            lines = DensitometricFeatureExtractor.count_lines(block)
            sentences = DensitometricFeatureExtractor.count_sentences(block)

            # extract link text
            link_text = AnchorUtil.extract_anchor_text(text_block.text)
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


basic_cleaner = Cleaner(forms=False, style=False, meta=False, page_structure=False, remove_unknown_tags=False, safe_attrs_only=False)

class HTMLLoader(object):
    XML_ENCODING_DECLARATION = re.compile(r'^\s*<\?xml[^>]*?>', re.I);

    @staticmethod
    def from_file(filename):
        html = read_unicode_from(filename)
        # lxml doesn't like xml encoding declaration in unicode html
        html = HTMLLoader.XML_ENCODING_DECLARATION.sub('', html)
        html = basic_cleaner.clean_html(html)
        return lxml.html.document_fromstring(html)


class MatrixScaler(object):
    """Scale features to [-1, 1] or [0, 1], as recommended by [Lin 2003]."""

    def scale_data(self, matrix):
        """Scale and remember scaling factors.
        @param matrix 2-D numpy array
        """
        self.scaling_factors = []
        for c in xrange(matrix.shape[1]):
            col = matrix[:,c]
            non_negative = np.all(col >= 0)
            if non_negative:
                _min, _max = col.min(), col.max()
                _range = _max-_min
                if _range > 0:
                    matrix[:,c] = (col-_min)/_range
                self.scaling_factors.append((non_negative, _min, _range))
            else:
                _max = np.abs(c).max()
                if _max > 0:
                    matrix[:,c] = col / _max
                self.scaling_factors.append((non_negative, _max))

        return matrix

    def scale(self, matrix):
        """Scale a new matrix.
        @param matrix 2-D numpy array as in scale_train
        """
        for c in xrange(matrix.shape[1]):
            col = matrix[:,c]
            config = self.scaling_factors[c]
            non_negative = config[0]
            if non_negative:
                _min, _max = config[1:]
                _range = _max-_min
                if _range > 0:
                    matrix[:,c] = (col-_min)/_range
            else:
                _max = self.scaling_factors[1]
                if _max > 0:
                    matrix[:,c] = col / _max

        return matrix


class ContentExtractionModel(object):
    def __init__(self, feature_extractors):
        self.feature_extractors = feature_extractors
        self.scaler = MatrixScaler()

    def extract_features(self, documents, unique=False):
        """@return numpy array of labels and features."""
        block_lists = [doc.get_blocks() for doc in documents]
        n_blocks = sum(len(blist) for blist in block_lists)
        n_features = sum(fe.n_features for fe in self.feature_extractors)

        features = np.zeros((n_blocks, n_features))
        labels = np.zeros(n_blocks)
        row = 0
        for blist in block_lists:
            end_row = row + len(blist)
            labels[row:end_row] = [block.label for block in blist]

            col = 0
            for fe in self.feature_extractors:
                end_col = col + fe.n_features
                features[row:end_row, col:end_col] = fe.extract(blist)
                col = end_col

            row = end_row

        return labels, features

    def train(self, documents, unique=False):
        labels, features = self.extract_features(documents, unique)
        scaled_features = self.scaler.scale_data(features)

        self.svm = svm.SVC()
        self.svm.fit(scaled_features, labels)

        print '>>> Training is over'

    def predict(self, document):
        _, features = self.extract_features([document])
        return self.svm.predict(self.scaler.scale(features))
