# Code from: http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Python
def LCS(X, Y):
    m = len(X)
    n = len(Y)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n+1) for i in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C

# from dragnet
def check_inclusion(x, y):
    """Given x, y (formatted as input to longest_common_subsequence)
    return a vector v of True/False with length(x)
    where v[i] == True if x[i] is in the longest common subsequence with y"""
    if len(y) == 0:
        return [False] * len(x)

    c = LCS(x, y)

    i = len(x)
    j = len(y)
    ret = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and x[i-1] == y[j-1]:
            ret.append(True)
            i -= 1
            j -= 1
        else:
            if j > 0 and (i == 0 or c[i][j-1] >= c[i-1][j]):
                j -= 1
            elif i > 0 and (j == 0 or c[i][j-1] < c[i-1][j]):
                ret.append(False)
                i -= 1

    ret.reverse()
    return ret
