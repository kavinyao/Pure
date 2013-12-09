import sys
import random
import numpy as np
from argparse import ArgumentParser
from sklearn import svm, cross_validation
from readability import L3SDocumentLoader, DragnetDocumentLoader, DensitometricFeatureExtractor, Evaluator

class MatrixScaler(object):
    """Scale features to [-1, 1] or [0, 1], as recommended by [Lin 2003]."""

    def scale_train(self, matrix):
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


def train_model(documents):
    features = []
    labels = []
    for doc in documents:
        doc_features, doc_labels = doc.get_training_example(DensitometricFeatureExtractor)
        features.extend(doc_features)
        labels.extend(doc_labels)

    print '#Examples:', len(labels)
    print '#Positive:', labels.count(1)

    data = np.array(features)
    target = np.array(labels)

    scaler = MatrixScaler()
    scaled_data = scaler.scale_train(data)

    clf = svm.SVC()
    clf.fit(scaled_data, labels)

    return clf, scaler


if __name__ == '__main__':
    tasks = ['cv', 'dump', 'evaluate', 'plot']

    parser = ArgumentParser(description='Run content extraction jobs.')
    parser.add_argument('dataset_dir', metavar='<Dataset Directory>')
    parser.add_argument('-t', '--task', help='task to run [%s]' % ('|'.join(tasks)))
    parser.add_argument('-d', '--dataset', default='L3S', help='set which dataset to use (L3S or Draget)')
    parser.add_argument('-l', '--limit', type=int, default=0, help='maximum number of documents to use, default to all')
    parser.add_argument('-s', '--scoring', default=None, help='scoring method of cross validation')
    parser.add_argument('-u', '--unique', action='store_true', default=False, help='use unique feature-label combinations as examples')
    parser.add_argument('-o', '--output', default='output.data', help='specify output file of various tasks')
    parser.add_argument('-r', '--ratio', default=0.7, type=float, help='when testing, ratio of examples as training data')

    args = parser.parse_args()

    if args.task not in tasks:
        print 'No task specified.'
        sys.exit(1)

    loader = L3SDocumentLoader(args.dataset_dir) if args.dataset == 'L3S' else DragnetDocumentLoader(args.dataset_dir)
    documents = loader.get_documents(args.limit)
    training_documents = documents if args.task != 'evaluate' else documents[:int(len(documents)*args.ratio)]

    raw_features = []
    raw_labels = []
    for doc in training_documents:
        doc_features, doc_labels = doc.get_training_example(DensitometricFeatureExtractor)
        raw_features.extend(doc_features)
        raw_labels.extend(doc_labels)

    if args.unique:
        unique_examples = set(t for t in zip(raw_labels, (tuple(fs) for fs in raw_features)))
        labels, features = [], []
        for t in unique_examples:
            labels.append(t[0])
            features.append(t[1])
    else:
        labels, features = raw_labels, raw_features

    assert len(raw_features[0]) == len(features[0])
    print '#Examples:', len(labels)
    print '#Positive:', labels.count(1)

    if args.task == 'cv':
        data = np.array(features)
        target = np.array(labels)

        scaler = MatrixScaler()
        scaled_data = scaler.scale_train(data)

        clf = svm.SVC()
        scores = cross_validation.cross_val_score(clf, scaled_data, target, cv=10, scoring=args.scoring)
        print '%s: %0.2f (+/- %0.2f)' % ((args.scoring or 'Precision').capitalize(), scores.mean(), scores.std()*2)
    elif args.task == 'evaluate' or args.task == 'plot':
        plot = args.task == 'plot'
        data = np.array(features)
        target = np.array(labels)

        scaler = MatrixScaler()
        scaled_data = scaler.scale_train(data)

        clf = svm.SVC()
        clf.fit(data, target)

        test_documents = documents if plot else documents[int(len(documents)*args.ratio):]
        evaluator = Evaluator(test_documents, DensitometricFeatureExtractor, clf, scaler)
        evaluator.evaluate()
        evaluator.report(args.output if plot else None)
    elif args.task == 'dump':
        with open(args.output, 'w') as output:
            for i in xrange(len(labels)):
                label, feature = labels[i], features[i]
                line = [str(label)]
                for j, f in enumerate(feature, 1):
                    line.append('%d:%f' % (j, f))
                output.write(' '.join(line)+'\n')
