cimport numpy as np

from rgr.graph.graph cimport Graph
from rgr.constants.types cimport DTYPE_ADJ

cpdef Graph complete_graph(int n_nodes)

cpdef Graph cycle_graph(int n_nodes)

cpdef Graph star_graph(int n_nodes)

cpdef Graph erdos_renyi_graph(int n_nodes, float prob_edge)

cpdef Graph peterson_graph()