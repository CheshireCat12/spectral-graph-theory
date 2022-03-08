import numpy as np
cimport numpy as np

from sgt.graph.graph cimport Graph


cdef class Complete(Graph):

    def __init__(self, int n_nodes):
        super().__init__(n_nodes)

    cpdef void __init_graph(self):
        self.adjacency = np.ones(self.n_nodes) - np.eye(self.n_nodes)
