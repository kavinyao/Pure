import sys
import numpy as np
from sklearn import svm, cross_validation
from readability import L3SDocumentLoader, DensitometricFeatureExtractor

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

    clf = svm.SVC()
    scores = cross_validation.cross_val_score(clf, data, target, cv=10, scoring='f1')
    print 'F1:%0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)
