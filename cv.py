import sys
import random
import numpy as np
from argparse import ArgumentParser
from sklearn import svm, cross_validation
from readability import L3SDocumentLoader, DragnetDocumentLoader
from readability import DensitometricFeatureExtractor, RelativePositionFeatureExtractor, IndicativeClassTokenFeatureExtractor
from readability import ContentExtractionModel, Evaluator, MatrixScaler
from readability import POSITIVE_LABEL, NEGATIVE_LABEL

if __name__ == '__main__':
    parser = ArgumentParser(description='Run content extraction jobs.')
    parser.add_argument('dataset_dir', metavar='<Dataset Directory>')
    parser.add_argument('-d', '--dataset', default='L3S', help='set which dataset to use (L3S or Draget)')
    parser.add_argument('-l', '--limit', type=int, default=0, help='maximum number of documents to use, default to all')
    parser.add_argument('-k', type=int, default=5, help='cross validation parameter K')

    parser.add_argument('-ft', '--shallow-text', action='store_true', default=False, help='use shallow text features')
    parser.add_argument('-fp', '--relative-position', action='store_true', default=False, help='use relative position features')
    parser.add_argument('-fi', '--indicative-token', action='store_true', default=False, help='use indicative token features')
    parser.add_argument('-tn', '--indicative-token-number', type=int, default=10, help='number of indicative tokens to use')

    args = parser.parse_args()

    feature_extractors = []
    if args.shallow_text:
        feature_extractors.append(DensitometricFeatureExtractor)
    if args.relative_position:
        feature_extractors.append(RelativePositionFeatureExtractor)
    if args.indicative_token:
        feature_extractors.append(IndicativeClassTokenFeatureExtractor(args.indicative_token_number))

    model = ContentExtractionModel(feature_extractors, unique=True)

    loader = L3SDocumentLoader(args.dataset_dir) if args.dataset == 'L3S' else DragnetDocumentLoader(args.dataset_dir)
    documents = loader.get_documents(args.limit)
    n_documents = len(documents)
    round_size = n_documents / args.k

    for i in range(args.k):
        print 'Round %d/%d...' % (i+1, args.k)
        test_documents = documents[i*round_size:(i+1)*round_size]
        training_documents = documents[:i*round_size] + documents[(i+1)*round_size:]

        model.train(training_documents)

        evaluator = Evaluator(test_documents, model)
        evaluator.evaluate()
        evaluator.report()
