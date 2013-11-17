import sys
import numpy as np
from sklearn import svm, cross_validation
from readability import L3SDocumentLoader, DensitometricFeatureExtractor

class MatrixScaler(object):
    """Scale features to [-1, 1] or [0, 1], as recommended by [Lin 2003]."""

    def scale(self, matrix):
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python %s <L3S base path> [limit]' % sys.argv[0]
        sys.exit(1)

    loader = L3SDocumentLoader(sys.argv[1])
    documents = loader.get_documents(0 if len(sys.argv) < 3 else int(sys.argv[2]))

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
    scaled_data = scaler.scale(data)

    clf = svm.SVC()
    scores = cross_validation.cross_val_score(clf, scaled_data, target, cv=10, scoring='f1')
    print 'F1:%0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)
