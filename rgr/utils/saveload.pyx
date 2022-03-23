import numpy as np
cimport numpy as np

from rgr.graph.graph cimport Graph

cpdef void save_graph(Graph graph, str filename):
    """
    Save the adjacency matrix of the graph as a numpy ´.npy´
    
    Args:
        graph (Graph): 
        filename (str): 

    Returns:
        void
    """
    with open(filename, 'wb') as file:
        np.save(file, graph.adjacency)


cpdef Graph load_graph(str filename):
    """
    Load the adjacency matrix from the given file and create the graph 

    Args:
        filename (str): 

    Returns:
        Graph - loaded graph
    """
    adjacency = np.load(filename)

    return Graph(adjacency)
