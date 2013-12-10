import sys
import random
import numpy as np
from argparse import ArgumentParser
from sklearn import svm, cross_validation
from readability import L3SDocumentLoader, DragnetDocumentLoader
from readability import DensitometricFeatureExtractor
from readability import ContentExtractionModel, Evaluator, MatrixScaler
from readability import POSITIVE_LABEL, NEGATIVE_LABEL

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

    model = ContentExtractionModel([DensitometricFeatureExtractor])

    if args.unique:
        # TODO
        raise NotImplementedError

    if args.task == 'cv':
        target, data = model.extract_features(training_documents)

        # gather label statistics
        positive_count = np.sum(target == POSITIVE_LABEL)
        print '#Examples:', len(target)
        print '#Positive:', positive_count

        scaler = MatrixScaler()
        scaled_data = scaler.scale_data(data)

        clf = svm.SVC()
        scores = cross_validation.cross_val_score(clf, scaled_data, target, cv=10, scoring=args.scoring)
        print '%s: %0.2f (+/- %0.2f)' % ((args.scoring or 'Precision').capitalize(), scores.mean(), scores.std()*2)
    elif args.task == 'evaluate' or args.task == 'plot':
        plot = args.task == 'plot'
        model.train(training_documents)

        test_documents = documents if plot else documents[int(len(documents)*args.ratio):]
        evaluator = Evaluator(test_documents, model)
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
