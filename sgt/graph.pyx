import numpy as np
cimport numpy as np

cdef class Graph:

    def __init__(self, int n_nodes):
        self.n_nodes = n_nodes
        self.adjacency = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)

    @property
    def n_nodes(self):
        return self.__n_nodes

    @property
    def adjacency(self):
        return self.__adjacency

    @n_nodes.setter
    def n_nodes(self, value):
        self.__n_nodes = value

    @adjacency.setter
    def adjacency(self, value):
        self.__adjacency = value

    cpdef int degree(self, int idx_node):
        """
        Compute the degree of the given node.
        Sum the row of the adjacency matrix at the index of the given node
        
        Args:
            idx_node: int 

        Returns:
            (int) degree of the given node

        """
        return np.sum(self.adjacency[idx_node, :])
