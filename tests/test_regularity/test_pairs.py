
import numpy as np
import pytest

from rgr.algorithms.partition_pair import PartitionPairSlow, PartitionPair
from rgr.algorithms.regularity import random_partition_init
from rgr.collection.standard import stochastic_block_model
from tests.test_code_external.graph_reducer.classes_pair import ClassesPair
from tests.test_regularity.utils import create_reg_szemeredi




@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (15, 3, 5, 0, 0),
                             (18, 3, 5, 0, 0),
                             (15, 3, 6, 0, 0),
                             (30, 5, 5, 0, 0),
                             (130, 5, 5, 0, 0),
                             (130, 7, 9, 0, 0),
                             (530, 5, 2, 0.1, 0.5),
                             (1503, 8, 2, 0.15, 0.3),
                             (1503, 8, 16, 0.15, 0.3),
                         ])
def test_pairs(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = create_reg_szemeredi(n_nodes, n_partitions, partitions)

    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPairSlow(np.asarray(graph.adjacency), partitions, r, s, eps=0.285)
            pair_expected = ClassesPair(np.asarray(graph.adjacency), reg.classes, r, s, epsilon=0.285)

            assert np.array_equal(pair.bip_adj, pair_expected.bip_adj_mat)
            assert np.array_equal(pair.r_indices, pair_expected.r_indices)
            assert np.array_equal(pair.s_indices, pair_expected.s_indices)
            assert pair.bip_avg_deg == pair_expected.bip_avg_deg
            assert pair.bip_density == pair_expected.bip_density
            assert pair.prts_size == pair_expected.classes_n

@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (15, 3, 5, 0, 0),
                             (18, 3, 5, 0, 0),
                             (15, 3, 6, 0, 0),
                             (30, 5, 5, 0, 0),
                             (130, 5, 5, 0, 0),
                             (130, 7, 9, 0, 0),
                             (530, 5, 2, 0.1, 0.5),
                             (1503, 8, 2, 0.15, 0.3),
                             (1503, 8, 16, 0.15, 0.3),
                         ])
def test_pairs_fast(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = create_reg_szemeredi(n_nodes, n_partitions, partitions)

    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPair(graph.adjacency, partitions, r, s, eps=0.285)
            pair_expected = ClassesPair(np.asarray(graph.adjacency), reg.classes, r, s, epsilon=0.285)

            assert np.array_equal(pair.r_indices, pair_expected.r_indices)
            assert np.array_equal(pair.s_indices, pair_expected.s_indices)
            assert np.array_equal(np.array(pair.bip_adj), pair_expected.bip_adj_mat)
            assert pair.prts_size == pair_expected.classes_n
            assert pair.bip_avg_deg == pair_expected.bip_avg_deg
            assert pair.bip_sum_edges == pair_expected.bip_adj_mat.sum()
            assert pair.bip_density == pair_expected.bip_density
