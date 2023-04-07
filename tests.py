import numpy as np
import gaussian_elimination as ge
import gauss_seidel as gs
import util as ut

import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)

# 1
# x = (1, 2, 3)
A1 = np.array([[1, 1, 1], [2, 3, 4], [1, -1, 2]], dtype=np.float64)
b1 = np.array([6, 20, 5], dtype=np.float64)

# 2
# x = (1, 1, -1)
A2 = np.array([[5, 1, 1], [3, 4, 1], [3, 3, 6]], dtype=np.float64)
b2 = np.array([5, 6, 0], dtype=np.float64)

# 3
# x = (-2, -14/3, 1, 16/3)
A3 = np.array([[4, -1, 8, 1], [3, -1, 5, 1], [2, 1, 6, 2], [5, -1, 7, 1]], dtype=np.float64)
b3 = np.array([10, 9, 8, 7], dtype=np.float64)

# 4
# x = (2, -1, 3)
# Good for LU
A4 = np.array([[1, -3, 2], [-2, 8, -1], [4, -6, 5]], dtype=np.float64)
b4 = np.array([11, -15, 29], dtype=np.float64)
