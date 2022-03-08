
cdef class Graph:

    cdef:
        int __n_nodes
        double[:, ::1] __adjacency

    cpdef void __init_graph(self) except *

    cpdef int degree(self, int idx_node)