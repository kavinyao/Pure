import os
import re
import math
import random
import lxml.html
import numpy as np
from sklearn import svm
from collections import Counter
from lxml.html.clean import Cleaner

from dragnet.lcs import check_inclusion
from logging import log, WARNING, CRITICAL
from util import extract_css_tokens, str_to_unicode

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

        log('Loading over...', CRITICAL)
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
    def __init__(self, doc, label, text, tokens):
        """
        @param label POSITIVE_LABEL or NEGATIVE_LABEL
        @param text text with possibly inserted anchor markers
        @param tokens Counter of tokens from id/class attribute
        """
        self.doc = doc
        self.label = label
        self.text = text
        self.css_tokens = tokens

    def __repr__(self):
        return 'TextBlock[%s..] from %s' % (self.text[:42].encode('utf-8'), self.doc)


class Document(object):
    """A generic HTML document for extraction."""
    BLANK_REGEX = re.compile(r'\s+')
    CSS_DEPTH = 5

    def __init__(self, original, annotated):
        """
        @param original path to original HTML
        @param annotated path to gold corresponding gold standard
        """
        self.original = original

        log('loading %s' % self)
        self.html_doc = HTMLLoader.from_file(original)
        with open(original) as orig:
            self.html_string = orig.read()
        self.main_content = self._get_main_content(annotated)

        # for collecting text blocks
        self._text_cache = []
        self._css_token_cache = []
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

        log('generating text blocks %s' % self)
        self._traverse(self.html_doc.body)
        self._text_blocks_generated = True

        return self._text_blocks

    def extract_article(self, block_classes):
        article_blocks = []
        for i in xrange(len(block_classes)):
            if block_classes[i] == POSITIVE_LABEL:
                article_blocks.append(AnchorUtil.remove_markers(self._text_blocks[i].text))

        if not article_blocks:
            log('[WARN]: nothing extracted from %s, returning all content' % self, WARNING)
            return self.normalize_html_text(self.html_doc.body.text_content())
        else:
            return self.normalize_html_text('\n'.join(article_blocks))

    def get_main_content(self):
        return self.main_content

    TAGS_TO_IGNORE = set('style,script,option,object,embed,applet,link,noscript'.split(','))
    # Note: <br> not considered inline as it modifies visual structure
    TAGS_INLINE = set('a,b,big,i,small,tt,abbr,acronym,cite,code,dfn,em,kbd,strong,samp,var,bdo,img,map,object,q,span,sub,sup,button,input,label,select,textarea'.split(','))

    def _traverse(self, elem, depth=0):
        if elem.tag in Document.TAGS_TO_IGNORE:
            return

        tokens = self._extract_css_tokens(elem)
        if tokens:
            self._css_token_cache.append(tokens)

        flush = elem.tag not in Document.TAGS_INLINE
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

        if tokens:
            self._css_token_cache.pop()

    def _extract_css_tokens(self, elem):
        class_id = elem.attrib.get('class', '') + ' ' + elem.attrib.get('id', '')
        return extract_css_tokens(class_id)

    def _generate_block(self):
        if not self._text_cache:
            return

        text = self.normalize_html_text(' '.join(self._text_cache))
        if text == '':
            self._text_cache = []
            return

        # TODO: a text block must be longer than 3 words to be eligible for testing as main content
        label = POSITIVE_LABEL if text.count(' ') > 2 and AnchorUtil.remove_markers(text) in self.main_content else NEGATIVE_LABEL
        css_tokens = Counter()
        for tokens in self._css_token_cache[-Document.CSS_DEPTH:]:
            css_tokens.update(tokens)
        self._text_blocks.append(Block(self, label, text, css_tokens))

        self._text_cache = []

    def normalize_html_text(self, text):
        return Document.BLANK_REGEX.sub(' ', text).strip()


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
        return str_to_unicode(content)


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

    def __init__(self, model):
        """
        @param docs list<Document> to test
        @param model an extraction model with predict method
        """
        self.model = model
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.lcs_precisions = []
        self.lcs_recalls = []
        self.lcs_f1s = []

    def evaluate(self, training_docs, test_docs):
        self.model.train(training_docs)

        precisions = []
        recalls = []
        lcs_precisions = []
        lcs_recalls = []

        for doc in test_docs:
            # TODO: ugly
            if self.model.predict_classes():
                classes = self.model.predict(doc)
                extracted_content = doc.extract_article(classes)
            else:
                extracted_content = self.model.predict(doc)
            main_content = doc.get_main_content()

            e_list = extracted_content.encode('utf-8').split()
            m_list = main_content.encode('utf-8').split()

            # do token-level comparison
            e_words = set(e_list)
            m_words = set(m_list)
            common_words = e_words.intersection(m_words)
            if not common_words:
                common_words = set(['CANNOT_BELIEVE_THIS'])
                log('WARN: no word predicted accurately for %s' % doc, WARNING)

            if len(e_words):
                precisions.append(1.0*len(common_words)/len(e_words))
            else:
                precisions.append(0.0)
            recalls.append(1.0*len(common_words)/len(m_words))

            # do common subsequence comparison
            flags = check_inclusion(e_list, m_list)
            n_common = sum(flags)
            if n_common == 0:
                n_common = 1
                log('WARN: no common sequence extracted for %s' % doc, WARNING)

            lcs_precisions.append(1.0*n_common/len(e_list))
            lcs_recalls.append(1.0*n_common/len(m_list))

        f_measures = [2/(1/p+1/r) for p,r in zip(precisions, recalls)]
        avg_prec = Evaluator.average(precisions)
        avg_rec = Evaluator.average(recalls)
        avg_f1 = Evaluator.average(f_measures)
        log('Average prec: %.4f' % avg_prec, CRITICAL)
        log('Average rec:  %.4f' % avg_rec, CRITICAL)
        log('Average F1:   %.4f' % avg_f1, CRITICAL)

        lcs_f_measures = [2/(1/p+1/r) for p,r in zip(lcs_precisions, lcs_recalls)]
        avg_lcs_prec = Evaluator.average(lcs_precisions)
        avg_lcs_rec = Evaluator.average(lcs_recalls)
        avg_lcs_f1 = Evaluator.average(lcs_f_measures)
        log('Average LCS prec: %.4f' % avg_lcs_prec, CRITICAL)
        log('Average LCS rec:  %.4f' % avg_lcs_rec, CRITICAL)
        log('Average LCS F1:   %.4f' % avg_lcs_f1, CRITICAL)

        self.precisions.append(avg_prec)
        self.recalls.append(avg_rec)
        self.f1s.append(avg_f1)
        self.lcs_precisions.append(avg_lcs_prec)
        self.lcs_recalls.append(avg_lcs_rec)
        self.lcs_f1s.append(avg_lcs_f1)

    def report(self):
        log('', CRITICAL)
        log('Final average prec: %.4f' % Evaluator.average(self.precisions), CRITICAL)
        log('Final average rec:  %.4f' % Evaluator.average(self.recalls), CRITICAL)
        log('Final average F1:   %.4f' % Evaluator.average(self.f1s), CRITICAL)

        log('Final average LCS prec: %.4f' % Evaluator.average(self.lcs_precisions), CRITICAL)
        log('Final average LCS rec:  %.4f' % Evaluator.average(self.lcs_recalls), CRITICAL)
        log('Final average LCS F1:   %.4f' % Evaluator.average(self.lcs_f1s), CRITICAL)

    @staticmethod
    def average(numbers):
        return 1.0 * sum(numbers) / len(numbers)


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


class RelativePositionFeatureExtractor(object):
    BUCKETS = 5
    n_features = 1

    @staticmethod
    def need_training():
        return False

    @staticmethod
    def extract(blocks):
        """
        @param blocks blocks of the same document, ordered by position
        """
        n = len(blocks)
        bucket_size = 1.0 * n / RelativePositionFeatureExtractor.BUCKETS
        # discretize
        features = [math.ceil(i/bucket_size) for i in range(1, n+1)]
        return np.array(features).reshape((n, 1))


class DensitometricFeatureExtractor(object):
    LINE_DELIMITERS = '.,?!'
    NUM_WORDS_CAP = 500
    NUM_WORDS_PER_SENTENCE_CAP = 50

    n_features = 7

    @staticmethod
    def need_training():
        return False

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

            num_words = 1.0*words
            num_words_per_sentence = 1.0*words/sentences
            num_words_per_line = 1.0*words/lines
            link_density = 1.0*link_words/words

            # regularization to avoid influence of outliers
            if num_words > DensitometricFeatureExtractor.NUM_WORDS_CAP:
                log('Warning: %s has more than %d words (%d), capping...' % (text_block, DensitometricFeatureExtractor.NUM_WORDS_CAP, num_words), WARNING)
                num_words = DensitometricFeatureExtractor.NUM_WORDS_CAP

            if num_words_per_sentence > DensitometricFeatureExtractor.NUM_WORDS_PER_SENTENCE_CAP:
                log('Warning: %s has more than %d words/sentence (%d), capping...' % (text_block, DensitometricFeatureExtractor.NUM_WORDS_PER_SENTENCE_CAP, num_words_per_sentence), WARNING)
                num_words_per_sentence = DensitometricFeatureExtractor.NUM_WORDS_PER_SENTENCE_CAP

            # number of words, average sentence length, text density, link density
            features.append([num_words, num_words_per_sentence, num_words_per_line, link_density])

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


class IndicativeClassTokenFeatureExtractor(object):
    """Learn css/id tokens from NB which are most indicative of non-content.
    Use the tokens as binary features."""

    def __init__(self, top_n=10):
        self.nb = NaiveBayesModel()
        self.n_features = top_n

    def need_training(self):
        return True

    def train(self, documents):
        self.nb.train(documents)
        most_negative = self.nb.most_weighted_negative_tokens(self.n_features)
        print most_negative
        if len(most_negative) != self.n_features:
            log('Warning: not enough negative tokens', WARNING)
        self.negative_tokens = [t for t, _ in most_negative]

    def extract(self, blocks):
        n_blocks = len(blocks)
        features = np.zeros((n_blocks, self.n_features))

        for r, block in enumerate(blocks):
            for c, token in enumerate(self.negative_tokens):
                if block.css_tokens[token] > 0:
                    features[r,c] = 1

        return features


basic_cleaner = Cleaner(forms=False, style=False, meta=False, page_structure=False, remove_unknown_tags=False, safe_attrs_only=False)

class HTMLLoader(object):
    XML_ENCODING_DECLARATION = re.compile(r'^\s*<\?xml[^>]*?>', re.I);

    @staticmethod
    def from_file(filename):
        html_string = read_unicode_from(filename)
        # lxml doesn't like xml encoding declaration in unicode html
        html = HTMLLoader.XML_ENCODING_DECLARATION.sub('', html_string)
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


def unique_rows(a):
    """http://stackoverflow.com/a/16971324/1240620
    @return 2-D array of unique rows of matrix.
    """
    b = a[np.lexsort(a.T)]
    return b[np.concatenate(([True], np.any(b[1:] != b[:-1], axis=1)))]


class ContentExtractionModel(object):
    def __init__(self, feature_extractors, unique):
        self.feature_extractors = feature_extractors
        self.scaler = MatrixScaler()
        self.unique = unique # whether use unique training examples only

    def extract_features(self, documents):
        """
        Note: if len(documents) == 1, it is assumed feature extraction is for prediction,
              under this scenario, unique mode will not be triggered.
        @return numpy array of labels and features.
        """
        batch_mode = len(documents) > 1

        block_lists = [doc.get_blocks() for doc in documents]
        n_blocks = sum(len(blist) for blist in block_lists)
        n_features = sum(fe.n_features for fe in self.feature_extractors)

        features = np.zeros((n_blocks, n_features))
        labels = np.zeros(n_blocks)
        row = 0
        # run feature extraction document by document as some feature extraction
        # may assume the blocks belongs to the same document
        for blist in block_lists:
            end_row = row + len(blist)
            labels[row:end_row] = [block.label for block in blist]

            col = 0
            for fe in self.feature_extractors:
                end_col = col + fe.n_features
                features[row:end_row, col:end_col] = fe.extract(blist)
                col = end_col

            row = end_row

        if batch_mode and self.unique:
            combined = np.hstack((labels.reshape((n_blocks, 1)), features))
            unique_examples = unique_rows(combined)
            labels = unique_examples[:, 0]
            features = unique_examples[:, 1:]

        if batch_mode:
            # gather label statistics
            positive_count = np.sum(labels == POSITIVE_LABEL)
            log('#Examples: %d' % len(labels), CRITICAL)
            log('#Positive: %d' % positive_count, CRITICAL)

        return labels, features

    def train(self, documents):
        # for feature extractor which need to know information about all documents
        # before extracting features
        for fe in self.feature_extractors:
            if fe.need_training():
                fe.train(documents)

        labels, features = self.extract_features(documents)
        scaled_features = self.scaler.scale_data(features)

        self.svm = svm.SVC()
        self.svm.fit(scaled_features, labels)

        log('>>> Training is over', CRITICAL)

    def predict_classes(self):
        return True

    def predict(self, document):
        _, features = self.extract_features([document])
        return self.svm.predict(self.scaler.scale(features))


class NaiveBayesModel(object):
    def train(self, documents):
        """Note: assumes that POSITIVE_LABEL = 1, NEGATIVE_LABEL = 0."""
        blocks = []
        for doc in documents:
            blocks.extend(doc.get_blocks())

        n_blocks = len(blocks)
        labels = np.array([block.label for block in blocks])
        self.positive_prob = np.sum(labels == POSITIVE_LABEL) / n_blocks

        all_tokens = set()
        for block in blocks:
            all_tokens.update(block.css_tokens.keys())

        n_tokens = len(all_tokens)
        # generate a deterministic mapping from token to indix
        token_list = list(all_tokens)
        indices = {token: i for i, token in enumerate(token_list)}
        train_matrix = np.zeros((n_blocks, n_tokens))

        for i, block in enumerate(blocks):
            for token, count in block.css_tokens.iteritems():
                train_matrix[i, indices[token]] = count

        log('>>> Sparseness: %.2f' % (1.0*np.sum(train_matrix == 0) / (n_blocks*n_tokens)), WARNING)

        self.token_list = token_list
        self.n_tokens = n_tokens
        self.indices = indices

        self.positive_counts = labels.dot(train_matrix) + 1;
        self.positive_probs = self.positive_counts / np.sum(self.positive_counts)
        self.negative_counts = (1-labels).dot(train_matrix) + 1;
        self.negative_probs = self.negative_counts / np.sum(self.negative_counts)

        log('>>> NB Training is over', CRITICAL)

    def most_indicative_tokens(self, n=10):
        p_to_n = np.log(self.positive_probs / self.negative_probs)
        indices = p_to_n.argsort()
        most_positive = tuple((self.token_list[i], p_to_n[i]) for i in indices[-n:][::-1])
        most_negative = tuple((self.token_list[i], p_to_n[i]) for i in indices[:n])
        return (most_positive, most_negative)

    def most_weighted_negative_tokens(self, n=10):
        n_to_p = np.log(self.negative_probs / self.positive_probs) * self.negative_counts
        indices = n_to_p.argsort()
        return tuple((self.token_list[i], n_to_p[i]) for i in indices[-n:][::-1])

    def predict(self, document):
        blocks = document.get_blocks()
        n_blocks = len(blocks)

        matrix = np.zeros((n_blocks, self.n_tokens))
        for i, block in enumerate(blocks):
            for token, count in block.css_tokens.iteritems():
                if token in self.indices:
                    matrix[i, self.indices[token]] = count
                else:
                    log('Warning: %s not in vocabulary' % token, WARNING)

        positive = np.log(self.positive_probs).dot(matrix.T) + np.log(self.positive_prob)
        negative = np.log(self.negative_probs).dot(matrix.T) + np.log(1-self.positive_prob)
        classes = positive >= negative
        return [POSITIVE_LABEL if cls else NEGATIVE_LABEL for cls in classes]
