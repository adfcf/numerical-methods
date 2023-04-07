import numpy as np
import util as ut

row_pivoting = True
print_iterations = True

# SISTEMA TESTE x = (1, 2, 3)
A = np.array([[1, 1, 1], [2, 3, 4], [1, -1, 2]], dtype=np.float64)
b = np.array([6, 20, 5], dtype=np.float64)

# Let A.x = b; if you consider x0 to be a solution of this LS, then the residual vector is how much A.x0 deviates from b.
def calculate_residual_vector(a: np.ndarray, x0: np.ndarray, b: np.ndarray):
    p = np.matmul(a, x0)
    rv = p - b;
    return rv

# Performs gaussian elimination on 'matrix'
def gaussian_eliminate(matrix: np.ndarray):

    if print_iterations:
        print('Original matrix:')
        print(matrix)

    SIZE = matrix.shape[0]

    for row in range(SIZE):
        
        # Parcial pivoting process
        if row_pivoting:

            leader_row_index = row + ut.abs_max_index(matrix[row:, row])

            temp = matrix[leader_row_index].copy()
            matrix[leader_row_index] = matrix[row]
            matrix[row] = temp

        matrix[row] /= matrix[row, row]

        # Nulifying elements below the pivot
        for row_ahead in range(row + 1, SIZE):
            matrix[row_ahead] -= matrix[row_ahead, row] * matrix[row]

        if print_iterations:
            print('Iteration:', row + 1)
            print(matrix)

# Returns a gaussian-eliminated copy of 'matrix'
# The argument isn't modified
def gaussian_elimination(matrix: np.ndarray):
    mat = matrix.copy()
    gaussian_eliminate(mat)
    return mat

# Takes a row-reduced matrix by a standard retro-substituting method.
# Returns a list of solutions
def find_solutions(ls: np.ndarray):
    SIZE = ls.shape[0]
    solutions = list([])
    for row in range(SIZE - 1, -1, -1):
        current_solution = ls[row, SIZE]
        for column in range(SIZE - 1, row, -1):
            current_solution -= (ls[row, column] * solutions[SIZE - column - 1])
        solutions.append(current_solution)
    solutions.reverse()
    return np.array(solutions, dtype=np.float64)