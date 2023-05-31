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
import splines as sp
import examples

import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
ge.print_iterations = False

samples = np.array([[0, 3], [0.5, 1.8616], [1.0, -0.5571], [1.5, -4.1987], [2.0, -9.0536]])
splin = sp.natural_cubic_spline(samples)
print(sp.calculate_value(splin[0], 0.25, 0.5))




