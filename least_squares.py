import numpy as np
import gaussian_elimination as ge
import util as ut

import matplotlib.pyplot as pp

# De acordo com Filho, um modo de se calcular o erro de uma regressao eh atraves do Quadrado Medio Residual 

# (squared) difference vector between the original samples and the least squares adjusted polynomial.
def residual_vector(samples: np.ndarray, polynomial: np.ndarray):
    rv = np.zeros(samples.shape[0], dtype=np.float64)
    for i in range(rv.size):
        squares = ut.calculate_squares(samples[i, 0], polynomial.size)
        rv[i] = (samples[i, 1] - (polynomial @ squares)) ** 2
    return rv

# a0 + a1x1 + a2x2
def residual_vector_v2(samples: np.ndarray, plane: np.ndarray):
    rv = np.zeros(samples.shape[0], dtype=np.float64)
    for i in range(rv.size):
        x = np.array([1, samples[i, 0], samples[i, 1]], dtype=np.float64)
        rv[i] = (samples[i, 2] - (plane @ x)) ** 2
    return rv

def residual_variance(rv: np.ndarray, number_of_parameters):
    return (rv.sum() / (rv.size - number_of_parameters))

def residual_variance_v2(rv: np.ndarray):
    return (rv.sum() / (rv.size - 3))

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

def linear_regression_2v(samples: np.ndarray):

    SIZE = samples.shape[0]
    lr_matrix = np.zeros((3, 4), dtype=np.float64)

    summation_x1 = samples[:, 0].sum()
    summation_x2 = samples[:, 1].sum()
    summation_x1x2 = (samples[:, 0] @ samples[:, 1]).sum()
    summation_x1x1 = (samples[:, 0] ** 2).sum()
    summation_x2x2 = (samples[:, 1] ** 2).sum()

    summation_y = samples[:, 2].sum()
    summation_x1y = (samples[:, 2] @ samples[:, 0]).sum()
    summation_x2y = (samples[:, 2] @ samples[:, 1]).sum()

    lr_matrix[0, 0] = SIZE
    lr_matrix[0, 1] = summation_x1
    lr_matrix[0, 2] = summation_x2

    lr_matrix[1, 0] = summation_x1
    lr_matrix[1, 1] = summation_x1x1
    lr_matrix[1, 2] = summation_x1x2

    lr_matrix[2, 0] = summation_x2
    lr_matrix[2, 1] = summation_x1x2
    lr_matrix[2, 2] = summation_x2x2
    
    lr_matrix[0, 3] = summation_y
    lr_matrix[1, 3] = summation_x1y
    lr_matrix[2, 3] = summation_x2y

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

def multiple_linear_regression(samples: np.ndarray):

    # Number of samples
    N = samples.shape[0]
    # Number of variables
    V = samples.shape[1]

    mlr_matrix = np.zeros((V, V + 1), dtype=np.float64)

    mlr_matrix[0, 0] = N
    for i in range(1, V):

        summation = samples[:, i - 1].sum()

        mlr_matrix[i, 0] = summation
        mlr_matrix[0, i] = summation

        for j in range(1, i + 1):

            summation =  (samples[:, i - 1] @ samples[:, j - 1]).sum()

            mlr_matrix[i, j] = summation
            mlr_matrix[j, i] = summation

    mlr_matrix[0, -1] = samples[:, -1].sum()
    for i in range(1, V):
        mlr_matrix[i, -1] = (samples[:, -1] @ samples[:, i - 1]).sum()

    print(mlr_matrix)

    ge.gaussian_eliminate(mlr_matrix)
    return ge.find_solutions(mlr_matrix)