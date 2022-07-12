import numpy as np


def trace_distance(vec1, vec2):
    mat1 = vector2matrix(vec1)
    mat2 = vector2matrix(vec2)
    dist = np.linalg.norm(mat1-mat2, 1)
    return dist


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(line.split()) + r'\\' for line in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


def vector2matrix(vec):
    size = len(vec) - 2
    matrix = np.zeros((size, size), dtype=complex)
    for i in range(size):
        matrix[i, i] = vec[i]

    matrix[1, 2] = vec[-2] + vec[-1]*1j
    matrix[2, 1] = matrix[1, 2].conjugate()
    return matrix
