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

# tati
samples = np.array([[1, 1], [2, 1.25], [3, 1.75], [4, 2.25], [5, 3], [6, 3.15]])

samples2 = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 1.0], [6.0, 3.0], [7.0, 3.0]])

samples3 = np.array([[-1.0, 0.86199], 
                     [-0.5, 0.95802], 
                     [0.0, 1.0986], 
                     [0.5, 1.29437]])

splin = sp.natural_cubic_spline(samples)

print(splin)

p0 = sp.to_standard_form(splin[0], 2)
p1 = sp.to_standard_form(splin[1], 2)
p2 = sp.to_standard_form(splin[2], 4)

print(sp.calculate_value(splin[0], 1.5, 2))
print(ut.image_of(p0, 1.5))


