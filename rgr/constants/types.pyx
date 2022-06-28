import numpy as np

DTYPE_ADJ = np.int8
DTYPE_STD = np.int32
DTYPE_IDX = np.uint32
DTYPE_UINT = np.uint32
DTYPE_FLOAT = np.float32

cpdef type get_dtype_adj():
    return DTYPE_ADJ

cpdef type get_dtype_idx():
    return DTYPE_IDX

cpdef type get_dtype_uint():
    return DTYPE_UINT
