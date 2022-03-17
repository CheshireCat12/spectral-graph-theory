import numpy as np
cimport numpy as np

from sgt.graph.graph cimport Graph
from sgt.constants.types cimport DTYPE_ADJ

cdef class Cycle(Graph):
    """
    Cycle graph class
    The node $n_i$ is linked to the node $n_{i+1}$
    except for node $n_{|V|-1} which is linked to node n_{0}

    |V| = n
    |E| = n
    """
    def __init__(self, int n_nodes):
        super().__init__(n_nodes)

    cpdef void __init_graph(self) except *:
        assert self.n_nodes >= 3, f'A cycle must contain at least 3 nodes'

        self.adjacency = np.sum([np.diag(np.ones(self.n_nodes - 1), offset) for offset in [-1, 1]],
                                axis=0,
                                dtype=DTYPE_ADJ)

        self.adjacency[0, -1] = 1
        self.adjacency[-1, 0] = 1
