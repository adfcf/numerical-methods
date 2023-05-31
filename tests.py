from math import e, log
import numpy as np
import gaussian_elimination as ge
import jacobi as jac
import gauss_seidel as gs
import cholesky_decomposition as ch
import least_squares as ls
import lagrange_interpolation as li
import power_method as pm
import util as ut
import newton_interpolation as ni
import polynomial_interpolation as pi
import examples

import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
ge.print_iterations = False

samples3 = np.array([[1.1, 1.5], [1.5, 2.1], [2.2, 2.3], [2.7, 3.5], [3.2, 4.2], [3.5, 5.4], [3.8, 7.0], [4.2, 8.3]])

samples3t = np.zeros((8, 3), dtype=np.float64)
samples3t[:, 0] = samples3[:, 0]
samples3t[:, 1] = samples3[:, 0] ** 2
samples3t[:, 2] = samples3[:, 1]

line = ls.multiple_linear_regression(samples3t)
print(line)




