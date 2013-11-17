import sys
from readability import L3SDocumentLoader, DensitometricFeatureExtractor

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python %s <L3S base path> [limit]' % sys.argv[0]
        sys.exit(1)

    loader = L3SDocumentLoader(sys.argv[1])
    documents = loader.get_documents(0 if len(sys.argv) < 3 else int(sys.argv[2]))
    for doc in documents:
        print doc.get_training_example(DensitometricFeatureExtractor)
