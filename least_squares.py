import numpy as np
import gaussian_elimination as ge
import util as ut

import matplotlib.pyplot as pp

# (squared) difference vector between the original samples and the least squares adjusted polynomial.
def residual_vector(samples: np.ndarray, polynomial: np.ndarray):
    rv = np.zeros(samples.shape[0], dtype=np.float64)
    for i in range(rv.size):
        squares = ut.calculate_squares(samples[i, 0], polynomial.size)
        rv[i] = (samples[i, 1] - (polynomial @ squares)) ** 2
    return rv

def residual_variance(rv: np.ndarray, number_of_parameters):
    return (rv.sum() / (rv.size - number_of_parameters))

def linear_regression(samples: np.ndarray):

    SIZE = samples.shape[0]
    lr_matrix = np.zeros((2, 3), dtype=np.float64)

    lr_matrix[0, 0] = SIZE

    summation = samples[:, 0].sum()
    lr_matrix[1, 0] = summation
    lr_matrix[0, 1] = summation

    lr_matrix[1, 1] = (samples[:, 0] ** 2).sum()
    lr_matrix[0, 2] = samples[:, 1].sum()
    lr_matrix[1, 2] = samples[:, 0] @ samples[:, 1]

    ge.gaussian_eliminate(lr_matrix)
    return ge.find_solutions(lr_matrix)

def polynomial_regression(samples: np.ndarray, degree):

    SIZE = degree + 1
    pr_matrix = np.zeros((SIZE, SIZE + 1), dtype=np.float64)

    x_summations = np.zeros((degree * 2) + 1, dtype=np.float64)
    for d in range(x_summations.size):
        x_summations[d] = (samples[:, 0] ** d).sum()

    y_summations = np.zeros(degree + 1, dtype=np.float64)
    for deg in range(y_summations.size):
        y_summations[deg] = (samples[:, 1] * (samples[:, 0] ** deg)).sum()

    # Setting coefficients partition
    for row in range(SIZE):
        for column in range(SIZE):
            pr_matrix[row, column] = x_summations[row + column]

    # Setting independent terms partition
    for i in range(y_summations.size):
        pr_matrix[i, -1] = y_summations[i]

    ge.gaussian_eliminate(pr_matrix)
    return ge.find_solutions(pr_matrix)
