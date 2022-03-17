import numpy as np
cimport numpy as np

from sgt.constants.types cimport DTYPE_ADJ

cdef class ErdosRenyi(Graph):

    def __init__(self, int n_nodes, float p):
        super().__init__(n_nodes)
        self.p = p

    cpdef void __init_graph(self):
        adj = np.random.choice([0, 1],
                               size=self.n_nodes**2,
                               p=[1-self.p, self.p]).reshape((self.n_nodes, self.n_nodes)).astype(DTYPE_ADJ)
        np.fill_diagonal(adj, 0)
        self.adjacency = adj
