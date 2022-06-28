import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_UINT_t sum_matrix(DTYPE_ADJ_t[:, ::1] arr):
    """
    Sum the entries of the given adjacency matrix.
    
    Args:
        arr: DTYPE_ADJ_t[:, ::1]

    Returns:
        DTYPE_UINT_t: sum of the given array

    """
    cdef:
        Py_ssize_t i, j
        Py_ssize_t max_i = arr.shape[0]
        Py_ssize_t max_j = arr.shape[1]
        DTYPE_UINT_t results = 0

    for i in range(max_i):
    # for x in prange(x_max, nogil=True):
        for j in range(max_j):
            results += arr[i, j]

    return results

def p_sum_matrix(arr):
    return sum_matrix(arr)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_UINT_t[::1] sum_mat_axis(DTYPE_ADJ_t[:, ::1] arr, int axis):
    """
    Sum the entries of the given adjacency matrix along the given axis.

    Args:
        arr: DTYPE_ADJ_t[:, ::1]
        axis: int

    Returns:
        DTYPE_UINT_t[::1]: sum of the given array

    """
    cdef:
        Py_ssize_t row, col
        Py_ssize_t x_max = arr.shape[0]
        Py_ssize_t y_max = arr.shape[1]
        DTYPE_UINT_t[::1] results


    if axis == 1:
        results = np.zeros(x_max, dtype=DTYPE_UINT)
    else:
        results = np.zeros(y_max, dtype=DTYPE_UINT)

    if axis == 1:
        for row in range(x_max):
        # for row in prange(x_max, nogil=True):
            for col in range(y_max):
                results[row] += arr[row, col]
    else:
        for col in range(y_max):
        # for col in prange(y_max, nogil=True):
            for row in range(x_max):
                results[col] += arr[row, col]

    return results

def p_sum_mat_axis(arr, axis):
    return sum_mat_axis(arr, axis)

