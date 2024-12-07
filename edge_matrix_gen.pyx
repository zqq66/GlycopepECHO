import cython
import numpy as np
from libc.math cimport fabs, exp
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def typological_sort_floyd_warshall(adjacency_matrix):
    cdef long n
    cdef long[:, ::1] dist_matrix_view,predecessors_view
    n = adjacency_matrix.shape[0]
    dist_matrix = adjacency_matrix.astype(int).copy()
    predecessors = np.zeros_like(adjacency_matrix)
    dist_matrix_view = dist_matrix
    predecessors_view = predecessors
    cdef long i, j, k
    for i in range(2,n):
        for j in range(n-i):
            for k in range(j+1,i+j):
                if dist_matrix_view[j,k] != 0 and dist_matrix_view[k,i+j] != 0:
                    if dist_matrix_view[j,i+j] < dist_matrix_view[j,k] + dist_matrix_view[k,i+j]:
                        dist_matrix_view[j,i+j] = dist_matrix_view[j,k] + dist_matrix_view[k,i+j]
                        predecessors_view[j,i+j] = k
    return dist_matrix, predecessors