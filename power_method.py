import numpy as np
import util as ut

MAX_ITERATIONS = 10
EPSILON = 0.0001

def power_method(A: np.ndarray):

    # TODO: check epsilon
    enough_accuracy = False

    current_x = A[0, :].copy()

    i = 0
    while i < MAX_ITERATIONS and not enough_accuracy:

        i += 1

        next_x = A @ current_x;

        next_x /= ut.euclidean_norm(next_x)
        current_x = next_x.copy()

    return current_x
