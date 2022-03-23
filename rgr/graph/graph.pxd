cimport numpy as np

cdef class Graph:

    cdef:
        np.ndarray __adjacency
