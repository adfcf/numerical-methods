import numpy as np
import util as ut

# a_matrix eh uma matriz quadrada de tamanho SIZE
# b_vector eh o vetor-coluna de termos independentes de tamanho SIZE
def gauss_seidel_method(a_matrix: np.ndarray, b_vector: np.ndarray):

    GAUSS_SEIDEL_MAX_ITERATIONS = 50
    GAUSS_SEIDEL_EPSILON = 0.05

    SIZE = a_matrix.shape[0]

    current_x = np.zeros(shape=(SIZE,), dtype=np.float64)
    inter_x = np.zeros(shape=(SIZE,), dtype=np.float64)
    next_x = np.zeros(shape=(SIZE,), dtype=np.float64)

    # Initial guess
    for i in range(SIZE):
        current_x[i] = b_vector[i] / a_matrix[i, i]

    # Ax = b
    # Divindo o SL por cada elemento da diagonal principal
    # Colocar pivotamento?
    pivot = 1.0
    for i in range(SIZE):
        pivot = a_matrix[i, i]
        a_matrix[i] /= pivot
        b_vector[i] /= pivot
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
        print('Iteration', iterations)
        for i in range(SIZE):
            print(current_x[i])
        print('Rel. Difference =', how_different)
        print('============================')

        if how_different <= GAUSS_SEIDEL_EPSILON:
            enough_accuracy = True

A = np.array([[1, 1, 1], [2, 3, 4], [1, -1, 2]], dtype=np.float64)
b = np.array([6, 20, 5], dtype=np.float64)

A2 = np.array([[5, 1, 1], [3, 4, 1], [3, 3, 6]], dtype=np.float64)
b2 = np.array([5, 6, 0], dtype=np.float64)

gauss_seidel_method(A2, b2)

