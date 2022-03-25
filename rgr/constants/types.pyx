import numpy as np

DTYPE_ADJ = np.int8
DTYPE_STD = np.int16
DTYPE_FLOAT = np.float32

cpdef type get_dtype_adj():
    return DTYPE_ADJ
