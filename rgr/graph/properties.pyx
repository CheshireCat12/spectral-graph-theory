import numpy as np


cpdef np.ndarray degrees(Graph graph):
    """
    
    Args:
        graph (Graph): 

    Returns:
        np.ndarray: the degree of all the nodes

    """
    return np.sum(graph.adjacency, axis=0)
