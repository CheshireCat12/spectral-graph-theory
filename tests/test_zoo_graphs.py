import numpy as np
import pytest
from sgt.constants.types import get_dtype_adj

from sgt.graph.complete import Complete
from sgt.graph.cycle import Cycle
from sgt.graph.graph import Graph

DTYPE_ADJ = get_dtype_adj()


def test_abstract_graph():
    with pytest.raises(NotImplementedError):
        Graph(n_nodes=5)


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
    graph = Complete(n_nodes=n_nodes)

    assert np.array_equal(graph.adjacency, expected_adj)


##############################
# Cycle Graph
##############################
@pytest.mark.parametrize('n_nodes',
                         [1, 2])
def test_complete_assert(n_nodes):
    with pytest.raises(AssertionError):
        Cycle(n_nodes)


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
    graph = Cycle(n_nodes=n_nodes)

    assert np.array_equal(graph.adjacency, expected_adj)
#
# @pytest.fixture
# def my_graph():
#     return Graph(n_nodes=4)
#
# [[0, 1, 0, 1],\n       [1, 0, 1, 0],\n       [0, 1, 0, 1],\n       [1, 0, 1, 0]]
# def test_degree(my_graph):
#     assert my_graph.degree(0) == 3
