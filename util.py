import numpy as np
from math import fabs
from math import sqrt

import matplotlib.pyplot as pp

# A set of points and a function
def plot_samples_fn(samples: np.ndarray, fn=None):

    x_axis = []
    f_y_axis = []

    MIN = samples[:, 0].min()
    MAX = samples[:, 0].max()

    STEP = (MAX - MIN) / 100

    x = MIN
    while x <= MAX:
        x_axis.append(x)
        if fn != None:
            f_y_axis.append(fn(x))
        x += STEP;

    if fn != None:
        pp.plot(x_axis, f_y_axis)

    pp.plot(samples[:, 0], samples[:, 1], 'rp')
    pp.show()

def calculate_squares(x, number_of_terms):
    squares = np.zeros(number_of_terms, dtype=np.float64)
    for i in range(number_of_terms):
        squares[i] += pow(x, i)
    return squares

def image_of(polynomial: np.ndarray, argument):
    return polynomial @ calculate_squares(argument, polynomial.size)

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

def euclidean_norm(v: np.ndarray):
    sum = 0.0
    for i in range(v.size):
        sum += v[i] * v[i]
    return sqrt(sum)

# Returns A|b
def couple_Ab(A: np.ndarray, b: np.ndarray):
    return np.append(A, b.reshape(A.shape[0], 1), axis=1)

# Attempts to make the matrix diagonally dominant by swapping its rows
def main_diagonal_swap_top(a: np.ndarray):
    abs_index = 0
    SIZE = a.shape[0]
    for i in range(SIZE - 1, 0, -1):
        abs_index = abs_max_index(a[:i - 1, i])
        if abs(a[i, i]) < abs(a[abs_index, i]):
            temp = a[i].copy()
            a[i] = a[abs_index].copy()
            a[abs_index] = temp


def main_diagonal_swap_down(a: np.ndarray):
    abs_index = 0
    SIZE = a.shape[0]
    for i in range(SIZE - 1):
        abs_index = i + 1 + abs_max_index(a[i + 1:, i])
        if abs(a[i, i]) < abs(a[abs_index, i]):
            temp = a[i].copy()
            a[i] = a[abs_index].copy()
            a[abs_index] = temp

# Returns an epsilon from a number 'n' of wanted significant digits.
def epsilon_from_significant_digits(n):
    return 0.5 * pow(10.0, -n)

def hilbert(i, j):
    return 1.0 / (i + j + 1.0)

def identity(i, j):
    return 1.0 if i == j else 0.0

def diagonal_product(A: np.ndarray):
    SIZE = A.shape[0]
    p = 1.0
    for i in range(SIZE):
        p *= A[i, i]
    return p

def iterative_test(ls: np.ndarray):

    # next_x = (A + I).current_x - b

    ls = ls.copy()

    MAX = 50
    EPSILON = 0.0001

    vector_b = ls[:, -1]
    matrix_a = ls[:, :-1]
    M = matrix_a + identity_matrix(ls.shape[0])

    current = vector_b.copy()
    next = vector_b.copy()

    i = 0
    while i < MAX:

        i += 1
        next = M @ current
        next -= vector_b

        diff = next - current;
        inf_norm = infinity_norm(diff) / infinity_norm(next)

        current = next.copy()

        if inf_norm <= EPSILON:
            break

    print(current)

