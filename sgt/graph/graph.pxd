cimport numpy as np

cdef class Graph:

    cdef:
        int __n_nodes
        np.ndarray __adjacency

    cpdef void __init_graph(self) except *

    cpdef int degree(self, int idx_node)