import numpy as np
import pytest

from rgr.algorithms.conditions.regularity_conditions import RegularityConditions
from rgr.algorithms.partition_pair import PartitionPair, PartitionPairSlow
from rgr.algorithms.regularity import random_partition_init
from rgr.collection.standard import stochastic_block_model
from tests.test_code_external.graph_reducer.classes_pair import ClassesPair
from tests.test_code_external.graph_reducer.conditions import alon1, alon2, alon3
from tests.test_regularity.utils import create_reg_szemeredi


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (30, 3, 5, 0, 0),
                             (30, 3, 5, 0.1, 0.1),
                             (30, 3, 5, 0.5, 0.5),
                             (18, 3, 5, 0, 0),
                             (18, 3, 5, 0.1, 0.1),
                             (15, 3, 6, 0, 0),
                             (30, 5, 5, 0, 0),
                             (71, 7, 7, 0, 0),
                             (130, 5, 5, 0, 0),
                             (130, 7, 9, 0, 0),
                             (130, 7, 9, 0.1, 0.2),
                             (530, 5, 2, 0.1, 0.5),
                             (1503, 8, 2, 0.15, 0.3),
                             (1503, 8, 16, 0.15, 0.3),
                         ])
def test_conditions(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = create_reg_szemeredi(n_nodes, n_partitions, partitions)
    reg.adj_mat = np.asarray(graph.adjacency, dtype=np.int32)

    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            # pair = PartitionPairSlow(np.asarray(graph.adjacency), partitions, r, s, eps=0.285)
            pair = PartitionPair(graph.adjacency, partitions, r, s, eps=0.285)
            pair_expected = ClassesPair(reg.adj_mat, reg.classes, r, s, epsilon=0.285)

            reg_cond = RegularityConditions(pair)

            expected_conditions = [alon1, alon2, alon3]
            conditions = [reg_cond.condition_1, reg_cond.condition_2, reg_cond.condition_3]
            for expected_condition, condition in zip(expected_conditions, conditions):

                expected_verified, expected_certs, expected_compls = expected_condition(reg, pair_expected)
                expected_cert_r, expected_cert_s = expected_certs
                expected_compl_r, expected_compl_s = expected_compls

                verified, certs_compls = condition()

                assert expected_verified == verified
                assert np.array_equal(expected_cert_r, certs_compls.r_certs)
                assert np.array_equal(expected_cert_s, certs_compls.s_certs)
                assert np.array_equal(expected_compl_r, certs_compls.r_compls)
                assert np.array_equal(expected_compl_s, certs_compls.s_compls)
