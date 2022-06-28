from rgr.constants.types cimport DTYPE_ADJ_t, DTYPE_UINT_t, DTYPE_UINT

cdef DTYPE_UINT_t[::1] sum_mat_axis(DTYPE_ADJ_t[:, ::1] arr, int axis)
cdef DTYPE_UINT_t sum_matrix(DTYPE_ADJ_t[:, ::1] arr)
