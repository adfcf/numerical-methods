import numpy as np
import util as ut

row_pivoting = True
print_results = True

# Does not modify its argument
# Returns a dictionary containing the matrices 'lower', 'upper', 'swap', along with the number of swap 'num_swaps'.
def lu_decompose(mat: np.ndarray):

    p_swaps = 0
    
    matrix = mat.copy()
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
        for row_ahead in range(iteration + 1, SIZE):
            multiplier = matrix[row_ahead, iteration] / matrix[iteration, iteration]
            matrix[row_ahead] -= multiplier * matrix[iteration]
            matrix_L[row_ahead, iteration] = multiplier

    if print_results:
        print('P Matrix')
        print(matrix_P)
        print('U Matrix')
        print(matrix)
        print('L Matrix')
        print(matrix_L)
        print('Swaps:', p_swaps)

    lu_results = dict()
    lu_results['lower'] = matrix_L
    lu_results['upper'] = matrix
    lu_results['swaps'] = matrix_P
    lu_results['num_swaps'] = p_swaps

    return lu_results