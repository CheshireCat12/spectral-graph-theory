from sgt.graph.graph import Graph
import pytest


@pytest.fixture
def my_graph():
    return Graph(n_nodes=4)


def test_degree(my_graph):
    assert my_graph.degree(0) == 3
