
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

from rgr.graph.graph cimport Graph

cpdef void graph_2_pyvis(graph: nx.Graph, str filename):
    """
    
    Args:
        graph (nx.Graph): 
        filename: 

    Returns:

    """
    nt = Network('1000px', '1000px')

    nt.from_nx(graph)
    nt.show_buttons(filter_=['physics'])
    nt.save_graph(filename)

cpdef void graph_2_img(Graph graph, str filename):
    plt.imsave(filename, graph.adjacency)