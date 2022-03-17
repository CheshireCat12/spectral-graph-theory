cdef class Synthetic(Graph):

    def __init__(self, int n_nodes, int n_clusters):
        super().__init__(n_nodes)

        self.n_clusters = n_clusters

    cpdef void __init_graph(self) except *:
        pass
