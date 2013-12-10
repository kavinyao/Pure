import sys

def print_pairs(pairs):
    for w, p in pairs:
        print '%.4f - %s' % (p, w)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: python %s <doc_root> <limit> <n>' % sys.argv[0]
        sys.exit(1)

    from readability import DragnetDocumentLoader, NaiveBayesModel

    loader = DragnetDocumentLoader(sys.argv[1])
    docs = loader.get_documents(limit=int(sys.argv[2]))

    nb = NaiveBayesModel()
    nb.train(docs)

    print 'Number of tokens:', nb.n_tokens

    most_indicative = nb.most_indicative_tokens(int(sys.argv[3]))
    print 'Most positive:'
    print_pairs(most_indicative[0])
    print 'Most negative:'
    print_pairs(most_indicative[1])
