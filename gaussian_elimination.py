import numpy as np
import util as ut

row_pivoting = True

def gaussian_elimination(matrix: np.ndarray):

    print('Original matrix')
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
        for nulifying_row in range(row + 1, SIZE):
            matrix[nulifying_row] -= matrix[nulifying_row, row] * matrix[row]

        print('Iteration', row + 1)
        print(matrix)


def gaussian_elimination_with_lp(matrix: np.ndarray):

    print('Original matrix')
    print(matrix)

    p_swaps = 0

    SIZE = matrix.shape[0]
    matrix_P = ut.identity_matrix(SIZE)
    matrix_L = ut.identity_matrix(SIZE)

    for iteration in range(SIZE):

        multiplier = 1.0

        # Pivoting process
        if row_pivoting:

            leader_row_index = iteration + ut.abs_max_index(matrix[iteration:, iteration])

            temp = matrix[leader_row_index].copy()
            matrix[leader_row_index] = matrix[iteration]
            matrix[iteration] = temp

            temp = matrix_P[leader_row_index].copy()
            matrix_P[leader_row_index] = matrix_P[iteration]
            matrix_P[iteration] = temp

            p_swaps += 1

        # Nulifying elements below the pivot
        for nulifying_row in range(iteration + 1, SIZE):
            multiplier = matrix[nulifying_row, iteration] / matrix[iteration, iteration]
            matrix[nulifying_row] -= multiplier * matrix[iteration]
            matrix_L[nulifying_row, iteration] = multiplier

        print('Iteration', iteration + 1)
        print(matrix)

    print('P')
    print(matrix_P)
    print('L')
    print(matrix_L)
    print('Swaps:', p_swaps)

def find_solutions(ls: np.ndarray):
    gaussian_elimination(ls)
    SIZE = ls.shape[0]
    solutions = list([])
    for row in range(SIZE - 1, -1, -1):
        current_solution = ls[row, SIZE]
        for column in range(SIZE - 1, row, -1):
            current_solution -= (ls[row, column] * solutions[SIZE - column - 1])
        solutions.append(current_solution)
    solutions.reverse()
    return solutions