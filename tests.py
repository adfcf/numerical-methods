import numpy as np
import gaussian_elimination as ge
import jacobi as jac
import gauss_seidel as gs
import cholesky_decomposition as ch
import power_method as pm
import util as ut
import newton_interpolation as ni
import examples

import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
ge.print_iterations = False

A8 = np.array([[0.0, 3], [0.2, 2.760], [0.3, 2.655], [0.5, 2.625], [0.7, 3.035], [1.0, 5.000]], dtype=np.float64)

print(ni.div_difference(A8, 1));