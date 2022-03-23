import numpy as np
import pytest
import os
from rgr.collection.standard import complete_graph
from rgr.graph.graph import Graph
from rgr.utils.saveload import save_graph, load_graph

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           './test_data/')


def test_save_complete(tmpdir):
    test_filename = tmpdir.join('complete_graph.npy')
    # test_filename = os.path.join(FOLDER_DATA, 'complete_10.npy')
    # print(test_filename)
    graph = complete_graph(10)

    save_graph(graph, str(test_filename))

    adj = np.load(str(test_filename))

    assert np.array_equal(adj, graph.adjacency)


def test_load_complete():
    graph = complete_graph(10)

    graph_loaded = load_graph(os.path.join(FOLDER_DATA, 'complete_10.npy'))

    assert np.array_equal(graph.adjacency, graph_loaded.adjacency)


