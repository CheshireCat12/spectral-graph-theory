cimport numpy as np

ctypedef np.int8_t DTYPE_ADJ_t
ctypedef np.int32_t DTYPE_STD_t
ctypedef np.uint32_t DTYPE_IDX_t
ctypedef np.float32_t DTYPE_FLOAT_t

cdef:
    DTYPE_ADJ
    DTYPE_STD
    DTYPE_IDX
    DTYPE_FLOAT

cpdef type get_dtype_adj()
cpdef type get_dtype_idx()