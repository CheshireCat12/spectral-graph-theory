import networkx as nx

from rgr.graph.graph cimport Graph

cpdef graph_2_nx(Graph graph):
    """
    
    Args:
        graph (Graph): 

    Returns:

    """
    nx_graph = nx.Graph(graph.adjacency)

    return nx_graph