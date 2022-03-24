from rgr.graph.graph cimport Graph

cdef class Synthetic(Graph):
    cdef:
        int n_clusters
        int[::1] clusters