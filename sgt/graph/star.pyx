import numpy as np
cimport numpy as np

from sgt.graph.graph cimport Graph


cdef class Star(Graph):

    def __init__(self, int n_nodes):
        super().__init__(n_nodes)


    cpdef void __init_graph(self):
        cdef:
            double [:, ::1] adj = np.zeros((self.n_nodes, self.n_nodes))

        adj[0,1:] = 1
        adj[1:,0] = 1
        self.adjacency = adj
