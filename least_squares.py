import numpy as np
import gaussian_elimination as ge
import util as ut

import matplotlib.pyplot as pp

def residual_vector(samples: np.ndarray, polynomial: np.ndarray):
    rv = np.zeros(samples.shape[0], dtype=np.float64)
    for i in range(rv.size):
        squares = ut.calculate_squares(samples[i, 0], polynomial.size)
        rv[i] = samples[i, 1] - (polynomial @ squares)
    return rv

def residual_variance(rv: np.ndarray, number_of_parameters):
    return (rv.sum() / (rv.size - number_of_parameters))

def linear_regression(samples: np.ndarray):

    SIZE = samples.shape[0]
    lr_matrix = np.zeros((2, 4), dtype=np.float64)

    lr_matrix[0, 0] = SIZE

    summation = samples[:, 0].sum()
    lr_matrix[1, 0] = summation
    lr_matrix[0, 1] = summation

    lr_matrix[1, 1] = (samples[:, 0] ** 2).sum()
    lr_matrix[0, 2] = samples[:, 1].sum()
    lr_matrix[1, 2] = samples[:, 0] @ samples[:, 1]

    ge.gaussian_eliminate(lr_matrix)
    return ge.find_solutions(lr_matrix)

