from numpy import zeros
from numpy import float64

import util as ut
import gaussian_elimination as ge


def calculate_vandermonde_matrix(samples):

    # N = quantidade de amostras
    N = samples.shape[0]

    # A matriz é inicializada com zeros.
    vandermonde = zeros((N, N + 1), dtype=float64)

    # Na i-ésima iteração, calcula-se a i-ésima linha da matriz.
    for i in range(N):
        # Insere-se os N quadrados de x_i.
        vandermonde[i, :N] = ut.calculate_squares(samples[i, 0], number_of_terms=N)
        # Na última coluna é colocado o y_i.
        vandermonde[i, -1] = samples[i, 1]
    
    return vandermonde

def calculate_polynomial(samples):

    # Calcula a matriz de Vandermonde.
    vandermonde = calculate_vandermonde_matrix(samples)

    # Realiza a eliminação gaussiana sobre a matriz de Vandermonde.
    ge.gaussian_eliminate(vandermonde)

    # Encontra a solução do sistema a partir de sua matriz associada escalonada.
    solution = ge.find_solutions(vandermonde)

    return solution