import numpy as np
from math import fabs

# Index of the max (abs) element of 'array'
def abs_max_index(array: np.ndarray):
    max_value = 0.0
    max_index = 0
    index = 0
    for v in array:
        if fabs(v) > max_value:
            max_value = fabs(v)
            max_index = index
        index += 1
    return max_index

# The absolute value of the max element of 'vector'
def infinity_norm(vector: np.ndarray):
    return max(np.abs(vector.max()), np.abs(vector.min()))

# Linear equation format: (a0.x0 + a1.x1 + ... + an.xn = b) => (a0, a1, ..., an, b)
# Returns a matrix (SIZE x SIZE+1)
def ls_from_user_input():
    size = int(input('Size: '))
    equation_list = list([])
    for i in range(size):
        str_list_equation = input('Equation {0}: '.format(i + 1)).split(' ')
        equation_list.append(str_list_equation)
    return np.array(equation_list, dtype=np.float64)

# Returns identity matrix of size 'n'
def identity_matrix(n: int):
    identity = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        identity[i, i] = 1.0;
    return identity

# Returns A|b
def couple_Ab(A: np.ndarray, b: np.ndarray):
    return np.append(A, b.reshape(A.shape[0], 1), axis=1)

# Returns an epsilon from a number 'n' of wanted significant digits.
def epsilon_from_significant_digits(n):
    return 0.5 * pow(10.0, -n)