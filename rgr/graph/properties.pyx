import numpy as np


cpdef np.ndarray[DTYPE_STD_t, ndim=1] degrees(Graph graph):
    """
    Compute the degree of all the nodes
    
    Args:
        graph : Graph
             Graph to compute the degrees

    Returns:
        np.ndarray: the degree of all the nodes

    """
    return np.sum(graph.adjacency, axis=0)
