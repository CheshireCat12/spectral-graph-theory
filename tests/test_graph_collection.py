import numpy as np
import pytest
from rgr.constants.types import get_dtype_adj

from rgr.collection.standard import complete_graph, cycle_graph, star_graph, path_graph, erdos_renyi_graph, block_model

DTYPE_ADJ = get_dtype_adj()


##############################
# Complete Graph
##############################
@pytest.mark.parametrize('n_nodes, expected_adj',
                         [
                             (1, np.array([[0]],
                                          dtype=DTYPE_ADJ)),
                             (2, np.array([[0, 1],
                                           [1, 0]],
                                          dtype=DTYPE_ADJ)),
                             (3, np.array([[0, 1, 1],
                                           [1, 0, 1],
                                           [1, 1, 0]],
                                          dtype=DTYPE_ADJ)),
                         ])
def test_complete(n_nodes, expected_adj):
    graph = complete_graph(n_nodes)

    assert np.array_equal(graph.adjacency, expected_adj)


##############################
# Cycle Graph
##############################
# @pytest.mark.parametrize('n_nodes',
#                          [1, 2])
# def test_complete_assert(n_nodes):
#     with pytest.raises(AssertionError):
#         Cycle(n_nodes)
#

@pytest.mark.parametrize('n_nodes, expected_adj',
                         [
                             (3, np.array([[0, 1, 1],
                                           [1, 0, 1],
                                           [1, 1, 0]],
                                          dtype=DTYPE_ADJ)),
                             (4, np.array([[0, 1, 0, 1],
                                           [1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [1, 0, 1, 0]],
                                          dtype=DTYPE_ADJ)),
                         ])
def test_cycle(n_nodes, expected_adj):
    graph = cycle_graph(n_nodes)

    assert np.array_equal(graph.adjacency, expected_adj)


##############################
# Path Graph
##############################

@pytest.mark.parametrize('n_nodes, expected_adj',
                         [
                             (2, np.array([[0, 1],
                                           [1, 0]],
                                          dtype=DTYPE_ADJ)),
                             (3, np.array([[0, 1, 0],
                                           [1, 0, 1],
                                           [0, 1, 0]],
                                          dtype=DTYPE_ADJ)),
                             (4, np.array([[0, 1, 0, 0],
                                           [1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0]],
                                          dtype=DTYPE_ADJ)),
                         ])
def test_path(n_nodes, expected_adj):
    graph = path_graph(n_nodes)

    assert np.array_equal(graph.adjacency, expected_adj)


##############################
# Star Graph
##############################
@pytest.mark.parametrize('n_nodes, expected_adj',
                         [
                             (1, np.array([[0]],
                                          dtype=DTYPE_ADJ)),
                             (2, np.array([[0, 1],
                                           [1, 0]],
                                          dtype=DTYPE_ADJ)),
                             (3, np.array([[0, 1, 1],
                                           [1, 0, 0],
                                           [1, 0, 0]],
                                          dtype=DTYPE_ADJ)),
                             (4, np.array([[0, 1, 1, 1],
                                           [1, 0, 0, 0],
                                           [1, 0, 0, 0],
                                           [1, 0, 0, 0]],
                                          dtype=DTYPE_ADJ)),
                         ])
def test_star(n_nodes, expected_adj):
    graph = star_graph(n_nodes)

    assert np.array_equal(graph.adjacency, expected_adj)


##############################
# Erdos-renyi Graph
##############################
@pytest.mark.parametrize('n_nodes, expected_adj',
                         [
                             (4, np.array([[0, 0, 1, 1],
                                           [0, 0, 1, 1],
                                           [1, 1, 0, 0],
                                           [1, 1, 0, 0]],
                                          dtype=DTYPE_ADJ)),
                             (10, np.array([[0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
                                            [0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                                            [1, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                                            [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
                                            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                                            [1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                                            [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
                                            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
                                           dtype=DTYPE_ADJ)),
                         ])
def test_erdos_renyi(n_nodes, expected_adj):
    np.random.seed(42)
    graph = erdos_renyi_graph(n_nodes, prob_edge=0.8)

    assert np.array_equal(graph.adjacency, expected_adj)


def test_block_model():

    clusters = np.array([5] * 3)
    block_model(15, clusters, 0.1, 1, 0.1, 0)