import numpy as np

np.import_array()

cdef class Graph:
    """
    Abstract class of Graph

    Attributes
    ----------
    n_nodes : int
    adjacency : np.ndarray

    """

    def __init__(self, np.ndarray[DTYPE_ADJ_t, ndim=2] adjacency):
        self.adjacency = adjacency

    @property
    def adjacency(self):
        """np.ndarray[DTYPE_ADJ_t, ndim=2]: Adjacency matrix"""
        return np.array(self._adjacency)

    @property
    def n_nodes(self):
        """int: Number of nodes"""
        return self.adjacency.shape[0]

    @adjacency.setter
    def adjacency(self, np.ndarray[DTYPE_ADJ_t, ndim=2] adjacency):
        self._adjacency = adjacency
