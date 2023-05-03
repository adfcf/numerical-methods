import numpy as np
import util as ut

# Lagrange method for polynomial interpolation
# Samples: [[x1, y1], [x2, y2], ..., [xn, yn]]
# Let P(x) such that P(x1) = y1, P(x2) = y2, ..., P(xn) = yn.
# Returns P(argument)
def interpolate(samples: np.ndarray, argument: np.float64):

    SIZE = samples.shape[0]
    G = np.zeros((SIZE, SIZE), np.float64)

    for i in range(SIZE):
        for j in range(SIZE):

            xi = samples[i, 0]
            xj = samples[j, 0]

            if i == j:
                xi = argument

            G[i, j] = xi - xj

    summation = 0.0
    for i in range(SIZE):
        summation += samples[i, 1] / G[i].prod()

    return (ut.diagonal_product(G) * summation)