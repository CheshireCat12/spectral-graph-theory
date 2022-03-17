import numpy as np
cimport numpy as np

from sgt.graph.graph cimport Graph
from sgt.constants.types cimport DTYPE_ADJ

cdef class Complete(Graph):
    """
    Complete graph class
    All the nodes are linked to every other node

    |V| = n
    |E| = n(n-1)/2
    """

    def __init__(self, int n_nodes):
        super().__init__(n_nodes)

    cpdef void __init_graph(self):
        """
        Create the adjacency matrix of the complete graph
        Returns:

        """
        dim = (self.n_nodes, self.n_nodes)

        self.adjacency = np.ones(dim, dtype=DTYPE_ADJ)
        np.fill_diagonal(self.adjacency, 0)