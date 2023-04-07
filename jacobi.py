import numpy as np
import util as ut

print_iterations = False

def jacobi_method(a_matrix: np.ndarray, b_vector: np.ndarray):

    JACOBI_MAX_ITERATIONS = 50
    JACOBI_EPSILON = 0.05

    SIZE = a_matrix.shape[0]

    current_x = np.zeros(shape=(SIZE,), dtype=np.float64)
    next_x = np.zeros(shape=(SIZE,), dtype=np.float64)

    # Initial guess
    for i in range(SIZE):
        current_x[i] = b_vector[i] / a_matrix[i, i]


    # Preparing 'A' e 'b'
    pivot = 1.0
    for i in range(SIZE):
        pivot = a_matrix[i, i]
        a_matrix[i] /= pivot
        b_vector[i] /= pivot
        a_matrix[i, i] = 0.0

    # Stop conditions
    iterations = 0
    enough_accuracy = False

    while iterations < JACOBI_MAX_ITERATIONS and not enough_accuracy:

        # Calculates next solution
        next_x = b_vector.copy();
        next_x -= np.matmul(a_matrix, current_x)

        # Calculates progress by doing a comparison with the last solution
        x_difference = next_x - current_x
        how_different = ut.infinity_norm(x_difference) / ut.infinity_norm(next_x)

        current_x = next_x.copy()

        iterations += 1

        # Info
        if print_iterations:
            print('Iteration', iterations)
            for i in range(SIZE):
                print(current_x[i])
            print('Rel. Difference =', how_different)
            print('============================')

        if how_different <= JACOBI_EPSILON:
            enough_accuracy = True

        return current_x