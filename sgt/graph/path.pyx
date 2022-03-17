import numpy as np
cimport numpy as np

from sgt.graph.graph cimport Graph


cdef class Complete(Graph):

    def __init__(self, int n_nodes):
        super().__init__(n_nodes)

    cpdef void __init_graph(self):
        self.adjacency = np.sum([np.diag(np.ones(self.n_nodes), offset) for offset in [-1, 1]], axis=0)
