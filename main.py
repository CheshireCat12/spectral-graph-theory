import pstats

import numpy as np

# from rgr.algorithms.regularity import degrees, random_partition_init, classes_pair
from rgr.collection.standard import *
from rgr.utils.visualization import graph_2_pyvis, graph_2_img
from rgr.graph.graph import Graph
from rgr.graph.properties import degrees
#
# def profiler():
#     import cProfile
#     graph = complete_graph(10000)
#     profiler_ = cProfile.Profile()
#     profiler_.enable()
#     degrees(graph)
#     profiler_.disable()
#     stats = pstats.Stats(profiler_).sort_stats('time')
#     stats.print_stats()

import networkx as nx
# def run():
#     # graph = ErdosRenyi(512, 0.7)
#     graph = Cycle(16)
#     k = 4
#     eps = 1 / 4
#     classes = random_partition_init(n_nodes=graph.n_nodes,
#                                     n_classes=k)
#
#     for r in range(2, k+1):
#         for s in range(1, r):
#             classes_pair(graph.adjacency, classes, r, s, eps)
#             break
#         break

# from rgr.graph.erdos_renyi import ErdosRenyi
import sys

def check_size_objects():
   graph = complete_graph(10)
   adj = np.asarray(graph.adjacency)
   size = sys.getsizeof(graph.adjacency.base)
   print(size)
   print(adj.__sizeof__())
   print(adj.nbytes / 1024 / 1024)

   print(graph.n_nodes)
   print(graph.adjacency)

import networkx as nx
from rgr.utils.converter import graph_2_nx
def main():
    np.random.seed(45)

    # check_size_objects()
    # run()
    # print(graph.degree(0))
    # erdos_graph = ErdosRenyi(32, 0.8)
    # print(erdos_graph.adjacency)
    # r_graph = nx.erdos_renyi_graph(5, 0.8, seed=42)
    # print(r_graph)
    # print(nx.to_numpy_matrix(r_graph))
    size = 10
    p = 0.1
    random = nx.gnp_random_graph(size, p)

    clusters = np.array([100] * 10)

    graph = stochastic_block_model(100, 10, 0.1, 0.1, 0)

    nx_graph = graph_2_nx(graph)

    print(degrees(graph))
    np.random.seed(42)
    # graph_2_pyvis(nx_graph, 'test.html')
    graph_2_img(graph, 'test.png')
    np.random.seed(42)
    # graph_2_img(random_graph, 'test_nx.png')


    # profiler()



if __name__ == '__main__':
    main()
