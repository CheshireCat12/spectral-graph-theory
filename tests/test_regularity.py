from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from rgr.algorithms.conditions.regularity_conditions import RegularityConditions
from rgr.algorithms.partition_pair import PartitionPair
from rgr.algorithms.regularity import random_partition_init, check_regularity_pairs
from rgr.collection.standard import stochastic_block_model
from rgr.constants.types import get_dtype_idx
from tests.test_code_external.graph_reducer.classes_pair import ClassesPair
from tests.test_code_external.graph_reducer.conditions import alon1, alon2, alon3
from tests.test_code_external.graph_reducer.szemeredi_regularity_lemma import SzemerediRegularityLemma


@dataclass
class MockSzemeredi:
    N: int
    k: int
    epsilon: float
    classes: np.ndarray
    classes_cardinality: int


@pytest.mark.parametrize('n_nodes, n_partitions, expected',
                         [
                             (30, 5,
                              [np.array([], dtype=np.uint32),
                               np.array([2, 10, 13, 24, 26, 28], dtype=np.uint32),
                               np.array([5, 11, 16, 17, 22, 27], dtype=np.uint32),
                               np.array([1, 8, 14, 20, 23, 29], dtype=np.uint32),
                               np.array([4, 6, 7, 9, 18, 19], dtype=np.uint32),
                               np.array([0, 3, 12, 15, 21, 25], dtype=np.uint32)]
                              ),
                             (33, 5,
                              [np.array([11, 20, 24], dtype=np.uint32),
                               np.array([2, 10, 16, 17, 25, 26], dtype=np.uint32),
                               np.array([8, 13, 15, 23, 29, 32], dtype=np.uint32),
                               np.array([1, 5, 14, 22, 28, 30], dtype=np.uint32),
                               np.array([4, 6, 12, 18, 19, 21], dtype=np.uint32),
                               np.array([0, 3, 7, 9, 27, 31], dtype=np.uint32)]
                              ),
                             (163, 15,
                              [np.array([7, 26, 44, 54, 73, 90, 94, 100, 125, 131, 136, 149, 160], dtype=np.uint32),
                               np.array([33, 37, 51, 62, 80, 101, 119, 138, 142, 154], dtype=np.uint32),
                               np.array([8, 45, 55, 63, 89, 92, 93, 124, 146, 150], dtype=np.uint32),
                               np.array([16, 19, 24, 40, 56, 60, 66, 107, 118, 134], dtype=np.uint32),
                               np.array([22, 27, 61, 86, 108, 113, 121, 132, 144, 148], dtype=np.uint32),
                               np.array([2, 18, 30, 59, 71, 96, 97, 106, 110, 145], dtype=np.uint32),
                               np.array([10, 43, 74, 85, 98, 123, 130, 135, 147, 157], dtype=np.uint32),
                               np.array([13, 48, 49, 50, 64, 69, 83, 109, 111, 112], dtype=np.uint32),
                               np.array([3, 15, 20, 23, 52, 76, 78, 120, 122, 162], dtype=np.uint32),
                               np.array([6, 12, 14, 68, 75, 84, 95, 126, 139, 161], dtype=np.uint32),
                               np.array([0, 11, 35, 41, 46, 57, 91, 102, 152, 159], dtype=np.uint32),
                               np.array([1, 4, 17, 42, 65, 105, 116, 133, 141, 151], dtype=np.uint32),
                               np.array([5, 28, 34, 38, 53, 104, 114, 128, 129, 156], dtype=np.uint32),
                               np.array([29, 31, 32, 79, 82, 99, 115, 127, 137, 143], dtype=np.uint32),
                               np.array([25, 39, 58, 72, 77, 81, 140, 153, 155, 158], dtype=np.uint32),
                               np.array([9, 21, 36, 47, 67, 70, 87, 88, 103, 117], dtype=np.uint32)]
                              ),
                         ])
def test_partition_init(n_nodes, n_partitions, expected):
    np.random.seed(0)
    partitions = random_partition_init(n_nodes, n_partitions)

    for part_idx in range(0, n_partitions + 1):
        assert np.array_equal(partitions[part_idx], expected[part_idx])


def _init_szemeredi(n_nodes: int, n_partitions: int) -> MockSzemeredi:
    reg = MockSzemeredi(n_nodes,
                        n_partitions,
                        0.285,
                        np.empty((n_nodes,), dtype=get_dtype_idx()),
                        0)

    return reg


def _create_mock_partition(reg: MockSzemeredi, partitions: List[np.ndarray]) -> None:
    for c, indices in enumerate(partitions):
        reg.classes[indices] = c

    reg.classes_cardinality = partitions[1].size


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions',
                         [
                             (15, 3, 5),
                             (18, 3, 5),
                             (15, 3, 6),
                             (30, 5, 5),
                             (130, 5, 5),
                             (130, 7, 9),
                         ])
def test_pairs(n_nodes, n_blocks, n_partitions):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, 0, 0)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = _init_szemeredi(n_nodes, n_partitions)
    _create_mock_partition(reg, partitions)

    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPair(graph.adjacency, partitions, r, s, eps=0.285)
            pair_expected = ClassesPair(graph.adjacency, reg.classes, r, s, epsilon=0.285)

            assert np.array_equal(pair.bip_adj, pair_expected.bip_adj_mat)
            assert np.array_equal(pair.r_indices, pair_expected.r_indices)
            assert np.array_equal(pair.s_indices, pair_expected.s_indices)
            assert pair.bip_avg_deg == pair_expected.bip_avg_deg
            assert pair.bip_density == pair_expected.bip_density
            assert pair.prts_size == pair_expected.classes_n


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions',
                         [
                             (30, 3, 5),
                             (18, 3, 5),
                             (15, 3, 6),
                             (30, 5, 5),
                             (71, 7, 7),
                             (130, 5, 5),
                             (130, 7, 9),
                         ])
def test_conditions(n_nodes, n_blocks, n_partitions):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, 0, 0)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = _init_szemeredi(n_nodes, n_partitions)
    reg.adj_mat = graph.adjacency
    _create_mock_partition(reg, partitions)

    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPair(graph.adjacency, partitions, r, s, eps=0.285)
            pair_expected = ClassesPair(graph.adjacency, reg.classes, r, s, epsilon=0.285)

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


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions',
                         [
                             (30, 3, 5),
                             (18, 3, 5),
                             (15, 3, 6),
                             (30, 5, 5),
                             (71, 7, 7),
                             (130, 5, 5),
                             (130, 7, 9),
                         ])
def test_pairs_regularity(n_nodes, n_blocks, n_partitions):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, 0, 0)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = _init_szemeredi(n_nodes, n_partitions)
    reg.adj_mat = graph.adjacency
    _create_mock_partition(reg, partitions)

    reg.certs_compls_list = []
    reg.regularity_list = []
    reg.is_weighted = False
    reg.conditions = [alon1, alon2, alon3]

    n_irr_expected = SzemerediRegularityLemma.check_pairs_regularity(reg)

    tmp = check_regularity_pairs(graph.adjacency,
                                 partitions,
                                 epsilon=reg.epsilon)
    n_irregular_pairs, certificates_complements, regular_partitions = tmp

    assert n_irr_expected == n_irregular_pairs

    for r in range(2, n_partitions+1):
        for s in range(1, r):
            expected_cert_r, expected_cert_s = reg.certs_compls_list[r-2][s-1][0]
            expected_compl_r, expected_compl_s = reg.certs_compls_list[r-2][s-1][1]
            certs_compls = certificates_complements[r-2][s-1]

            assert np.array_equal(expected_cert_r, certs_compls.r_certs)
            assert np.array_equal(expected_cert_s, certs_compls.s_certs)
            assert np.array_equal(expected_compl_r, certs_compls.r_compls)
            assert np.array_equal(expected_compl_s, certs_compls.s_compls)
            assert np.array_equal(expected_compl_s, certs_compls.s_compls)
