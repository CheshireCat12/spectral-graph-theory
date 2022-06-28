cimport numpy as np
from rgr.constants.types cimport DTYPE_ADJ_t

cdef class Graph:

    cdef:
        readonly:
            int n_nodes
            DTYPE_ADJ_t[:, ::1] adjacency
