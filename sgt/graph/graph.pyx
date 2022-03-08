import numpy as np
cimport numpy as np


cdef class Graph:

    def __init__(self, int n_nodes):
        self.n_nodes = n_nodes
        self.adjacency = None

    @property
    def n_nodes(self):
        return self.__n_nodes

    @property
    def adjacency(self):
        if self.__adjacency is None:
            self.__init_graph()
        return self.__adjacency

    @n_nodes.setter
    def n_nodes(self, value):
        self.__n_nodes = value

    @adjacency.setter
    def adjacency(self, value):
        self.__adjacency = value

    cpdef void __init_graph(self) except *:
        """
        Abstract method that is used to create the graph.
        Each graph type (path, complete, ...) implements it own __init_graph()
        
        :return: 
        """
        raise NotImplementedError(f'Creator function for: "{self.__class__.__name__}" is not implemented!')

    cpdef int degree(self, int idx_node):
        """
        Compute the degree of the given node.
        Sum the row of the adjacency matrix at the index of the given node
        
        Args:
            idx_node: int 

        Returns:
            (int) degree of the given node

        """
        assert 0 <= idx_node < self.n_nodes, f'Idx node: {idx_node} not valid!'
        return np.sum(self.adjacency[idx_node, :])
