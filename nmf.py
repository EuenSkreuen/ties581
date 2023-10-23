# 1. https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/decomposition/_nmf.py#L1309
# 2. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
# 3. https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Algorithms

# TODO 1 : As per (3), try to write the algorithm so it does one iteration, and produces two new matrices
# TODO 2 : As per (2), try to write a function that has that specific objective function and iterates based on that (also check other possible objective functions)
# TODO 3 : As per (1), use the example matrix in github code, and compare results of your own 


import numpy as np
from sklearn.decomposition import NMF

def run_example_NMF():
    X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    print(f"Example matrix was following:\n{X}")
    print(f"W was:\n{W}")
    print(f"H was:\n{H}")
    V = np.matmul(W,H)
    print(f"Their multiplication V is:\n{V}")
    df = np.zeros(shape=(len(X),2))
    for i in range(0, len(X)):
        row = []
        for j in range(0, len(X[i])):
            row.append(X[i,j] - V[i,j])
        df[i] = row
    print(f"Difference matrix was:\n{df}")


def trace(A):
    # Trace of a matrix is equal to sum of its eigenvalues
    sum = 0
    for i in range(0, len(A)):
        sum += A[i,i]
    return sum


def frobenius_distance(X, Y):
    # TODO: calculate difference between matrices using Frobenius distance:
    # https://search.r-project.org/CRAN/refmans/SMFilter/html/FDist2.html
    # return np.sqrt(trace(np.matmul((),())))
    pass

run_example_NMF()
