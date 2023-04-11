import numpy as np
import util as ut

print_iterations = False
initial_guess = 'null'
swap_rows = 'down'

GAUSS_SEIDEL_MAX_ITERATIONS = 50
GAUSS_SEIDEL_EPSILON = 0.00001

def gauss_seidel_method(linear_system: np.ndarray):

    linear_system = linear_system.copy()
    SIZE = linear_system.shape[0]

    if swap_rows == 'down':
        ut.main_diagonal_swap_down(linear_system)
    elif swap_rows == 'top':
        ut.main_diagonal_swap_top(linear_system)

    a_matrix = linear_system[:, :-1]
    b_vector = linear_system[:, -1]

    current_x = np.zeros(shape=(SIZE,), dtype=np.float64)
    inter_x = np.zeros(shape=(SIZE,), dtype=np.float64)
    next_x = np.zeros(shape=(SIZE,), dtype=np.float64)

    # Initial guess
    for i in range(SIZE):
        current_x[i] = b_vector[i] / a_matrix[i, i]
        
    if type(initial_guess) == np.ndarray:
        current_x = initial_guess

    for i in range(SIZE):
        linear_system[i] /= a_matrix[i, i]
        a_matrix[i, i] = 0.0

    # Stop conditions
    iterations = 0
    enough_accuracy = False

    while iterations < GAUSS_SEIDEL_MAX_ITERATIONS and not enough_accuracy:

        # Calculates next solution
        inter_x = current_x.copy()
        next_x = b_vector.copy()
        for i in range(SIZE):
            next_x[i] -= np.matmul(a_matrix[i, :].transpose(), inter_x)
            inter_x[i] = next_x[i]

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

        enough_accuracy = how_different <= GAUSS_SEIDEL_EPSILON

    return current_x
