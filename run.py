import sys
import numpy as np
from argparse import ArgumentParser
from sklearn import svm, cross_validation
from readability import L3SDocumentLoader, DensitometricFeatureExtractor

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
    parser = ArgumentParser(description='Run content extraction jobs.')
    parser.add_argument('dataset_dir', metavar='<Dataset Directory>')
    parser.add_argument('-l', '--limit', type=int, default=0, help='maximum number of documents to use')
    parser.add_argument('-t', '--task', help='task to run [cv|dump]')
    parser.add_argument('-s', '--scoring', default=None, help='scoring method of cross validation')
    parser.add_argument('-u', '--unique', action='store_true', default=False, help='use unique feature-label combinations as examples')

    args = parser.parse_args()
    loader = L3SDocumentLoader(args.dataset_dir)
    documents = loader.get_documents(args.limit)

    raw_features = []
    raw_labels = []
    for doc in documents:
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
    elif args.task == 'dump':
        for i in xrange(len(labels)):
            label, feature = labels[i], features[i]
            line = [str(label)]
            for j, f in enumerate(feature, 1):
                line.append('%d:%f' % (j, f))
            print ','.join(line)
    else:
        print 'No task specified.'
