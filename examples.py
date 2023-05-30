import gaussian_elimination as ge
import jacobi as jac
import gauss_seidel as gs
import cholesky_decomposition as ch
import util as ut

import numpy as np

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

# 5
# x = (5, 1, -2)
# Good for JACOBI, GAUSS-SEIDEL
A5 = np.array([[10, 3, -2], [2, 8, -1], [1, 1, 5]], dtype=np.float64)
b5 = np.array([57, 20, -4], dtype=np.float64)

# 6
# Good for CHOLESKY
A6 = np.array([[4, -2, 2], [-2, 10, -7], [2, -7, 30]], dtype=np.float64)

# 7
# Good for LAGRANGE INTERPOLATION
A7 = np.array([[0.1, 1.221], [0.6, 3.320], [0.8, 4.953]], dtype=np.float64)

# 8
# Good for DIV DIFFERENCE
A8 = np.array([[0.0, 3], [0.2, 2.760], [0.3, 2.655], [0.5, 2.625], [0.7, 3.035], [1.0, 5.000]], dtype=np.float64)