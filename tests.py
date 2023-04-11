import numpy as np
import gaussian_elimination as ge
import jacobi as jac
import gauss_seidel as gs
import cholesky_decomposition as ch
import power_method as pm
import util as ut
import examples

import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
ge.print_iterations = False

print("POWER")
L = pm.power_method(np.array([[2.0, 3.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]]))
print(L)