import numpy as np


cdef class Graph:
    """
    Abstract class of Graph

    Attributes
    ----------
    n_nodes : int
    adjacency : np.ndarray

    """

    def __init__(self, np.ndarray adjacency):
        self.adjacency = adjacency

    @property
    def n_nodes(self):
        return self.adjacency.shape[0]

    @property
    def adjacency(self):
        return self.__adjacency

    @adjacency.setter
    def adjacency(self, value):
        self.__adjacency = value
