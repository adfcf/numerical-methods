import numpy as np
import util as ut
import gaussian_elimination as ge
from math import fabs

# Converts the format of the cubic spline
# From: (d, c, b, a) i.e. a(x - k)^3 + b(x - k)^2 + c(x - k) + d
# To:   (D, C, B, A) i.e. Ax^3 + Bx^2 + Cx + D
def to_standard_form(spline: np.ndarray, upper_bound: float):

    p = np.zeros(spline.size, dtype=np.float64)
    
    a = spline[3]
    b = spline[2]
    c = spline[1]
    d = spline[0]
    k = upper_bound
    ks = k * k
    kc = ks * k

    p[0] = d - (a * kc) + (b * ks) - (c * k)
    p[1] = (3 * a * ks) - (2 * b * k) + c
    p[2] = b - (3 * a * k)
    p[3] = a

    return p

def calculate_value(spline: np.ndarray, argument: float, upper_bound: float):
    value = 0.0
    power = 0
    for coefficient in spline:
        value += coefficient * ((argument - upper_bound) ** power)
        power += 1
    return value

def natural_cubic_spline(samples: np.ndarray):

    SIZE = samples.shape[0]
    N = SIZE - 1

    # [[a0, a1, a2, a3], [a0, a1, a2, a3], ..., n vezes]
    splines = np.zeros((N, 4), dtype=np.float64)

    x = samples[:, :1]
    y = samples[:, 1:]

    h = np.zeros((N,), np.float64)

    for i in range(N):
        h[i] = x[i + 1] - x[i]

    g = np.zeros((N + 1,), np.float64)

    for i in range(N - 1):

        a = (y[i + 2] - y[i + 1]) / h[i + 1]
        b = (y[i + 1] - y[i]) / h[i]

        g[i + 1] = 3 * (a - b)

    A = np.zeros((N + 1, N + 1), dtype=np.float64)
    A[0, 0] = 1
    A[N, N] = 1

    for i in range(N - 1):
        A[i + 1, i] = h[i]
        A[i + 1, i + 1] = 2 * (h[i] + h[i + 1])
        A[i + 1, i + 2] = h[i + 1]

    ls = ut.couple_Ab(A, g)
    ge.gaussian_eliminate(ls)
    solutions = ge.find_solutions(ls)

    for i in range(N):

        splines[i, 0] = y[i + 1]

        a = (y[i + 1] - y[i]) / h[i]
        b = (h[i] * (2 * solutions[i + 1] + solutions[i])) / 3

        splines[i, 1] = a + b
        splines[i, 2] = solutions[i + 1]
        splines[i, 3] = (solutions[i + 1] - solutions[i]) / (3 * h[i])

    return splines
