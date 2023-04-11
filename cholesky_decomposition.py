import numpy as np
from math import sqrt

def cholesky_decomposition(A: np.ndarray):

    SIZE = A.shape[0]
    L = np.zeros((SIZE, SIZE), np.float64)

    for i in range(SIZE):
        for j in range(i + 1):
            if i > j:
                summation = 0.0
                for k in range(j):
                    summation += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - summation) / L[j, j]
            else:
                summation = 0.0
                for k in range(j):
                    summation += L[j, k] * L[j, k]
                L[j, j] = sqrt(A[j, j] - summation)

    return L
