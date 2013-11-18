import sys
import numpy as np
from collections import Counter

def median(mylist):
    # thanks: http://stackoverflow.com/a/10482734/1240620
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python %s <plot.data> [plot_output.m]' % sys.argv[0]
        sys.exit(1)

    f1_scores = [float(l) for l in open(sys.argv[1])]
    mean = sum(f1_scores) / len(f1_scores)
    med = median(f1_scores)

    unique_values = sorted(set(f1_scores), reverse=True)
    counts = Counter(f1_scores)

    cumulative_counts = 0
    documents = []
    for val in unique_values:
        cumulative_counts += counts[val]
        documents.append(cumulative_counts)

    out_file = sys.argv[2] if len(sys.argv) > 2 else 'plot_output.m'
    with open(out_file, 'w') as out:
        out.write('X = [%s];\n' % ','.join('%d' % d for d in documents))
        out.write('Y = [%s];\n' % ','.join('%.4f' % s for s in unique_values))
        out.write("plot(X, Y, 'r', 'LineWidth', 2);\n")
        out.write("xlabel('# Documents');\n")
        out.write("ylabel('Token-Level F-Measure');\n")
        out.write("leg = legend('mean=%.2f%%;median=%.2f%%');\n" % (100*mean, 100*med))
        out.write("set(leg, 'Location', 'SouthWest');\n")
