import numpy as np
from math import fabs

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

def infinity_norm(vector: np.ndarray):
    return fabs(vector[abs_max_index(vector)])

# Linear equation format: a0x0 + a1x1 + ... + anxn = b => (a0, a1, ..., an, b)
# Returns a matrix (SIZE x SIZE+1)
def ls_from_user_input():
    size = int(input('Size: '))
    equation_list = list([])
    for i in range(size):
        str_list_equation = input('Equation {0}: '.format(i + 1)).split(' ')
        equation_list.append(str_list_equation)
    return np.array(equation_list, dtype=np.float64)

def identity_matrix(size: int):
    identity = np.zeros(shape=(size, size), dtype=np.float64)
    for i in range(size):
        identity[i, i] = 1.0;
    return identity