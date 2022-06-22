from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from rgr.algorithms.conditions.regularity_conditions import RegularityConditions
from rgr.algorithms.partition_pair import PartitionPair
from rgr.algorithms.refinement import Refinement
from rgr.algorithms.regularity import random_partition_init, check_regularity_pairs, is_partitioning_regular, regularity
from rgr.algorithms.matrix_reduction import matrix_reduction
from rgr.collection.standard import stochastic_block_model
from rgr.constants.types import get_dtype_idx, get_dtype_adj
from tests.test_code_external.graph_reducer.classes_pair import ClassesPair
from tests.test_code_external.graph_reducer.conditions import alon1, alon2, alon3
from tests.test_code_external.graph_reducer.szemeredi_regularity_lemma import SzemerediRegularityLemma
from tests.test_code_external.graph_reducer.refinement_step import indeg_guided, compute_indensities
from tests.test_code_external.graph_reducer.szemeredi_lemma_builder import generate_szemeredi_reg_lemma_implementation
from tests.test_code_external.graph_reducer.codec import Codec


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


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (15, 3, 5, 0, 0),
                             (18, 3, 5, 0, 0),
                             (15, 3, 6, 0, 0),
                             (30, 5, 5, 0, 0),
                             (130, 5, 5, 0, 0),
                             (130, 7, 9, 0, 0),
                             (530, 5, 2, 0.1, 0.5),
                         ])
def test_pairs(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
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
                         ])
def test_conditions(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
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


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (30, 3, 5, 0, 0),
                             (18, 3, 5, 0, 0),
                             (15, 3, 6, 0, 0),
                             (30, 5, 5, 0, 0),
                             (71, 7, 7, 0, 0),
                             (130, 5, 5, 0, 0),
                             (130, 7, 9, 0, 0),
                             (530, 5, 2, 0.1, 0.5),
                         ])
def test_pairs_regularity(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = _init_szemeredi(n_nodes, n_partitions)
    reg.adj_mat = graph.adjacency
    _create_mock_partition(reg, partitions)

    reg.certs_compls_list = []
    reg.regularity_list = []
    reg.is_weighted = False
    reg.conditions = [alon1, alon3, alon2]

    n_irr_expected = SzemerediRegularityLemma.check_pairs_regularity(reg)
    is_regular_expected = SzemerediRegularityLemma.check_partition_regularity(reg, n_irr_expected)

    ##########
    tmp = check_regularity_pairs(graph.adjacency,
                                 n_partitions,
                                 partitions,
                                 epsilon=reg.epsilon)
    n_irregular_pairs, certificates_complements, regular_partitions, sze_idx = tmp

    is_regular = is_partitioning_regular(n_irregular_pairs, n_partitions, reg.epsilon)

    assert n_irregular_pairs == n_irr_expected
    for exp_l1, l1 in zip(reg.certs_compls_list, certificates_complements):
        for exp_l2, l2 in zip(exp_l1, l1):
            assert np.array_equal(l2.r_certs, exp_l2[0][0])
            assert np.array_equal(l2.s_certs, exp_l2[0][1])
            assert np.array_equal(l2.r_compls, exp_l2[1][0])
            assert np.array_equal(l2.s_compls, exp_l2[1][1])
    assert all(val1 == val2 for val1, val2 in zip(reg.regularity_list, regular_partitions))
    # print(f'------ {reg.sze_idx}, {sze_idx}, {reg.sze_idx - sze_idx}')
    assert reg.sze_idx == sze_idx
    assert is_regular_expected == is_regular


from collections import defaultdict


def _reverse_partitions(classes):
    tmp = defaultdict(list)  # [] for _ in range(len(partitions))]
    for idx, cls in enumerate(classes):
        tmp[cls].append(idx)

    return sorted(tmp.items(), key=lambda x: x[0])


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             # (130, 5, 5, 0, 0),
                             # (130, 7, 9, 0, 0),
                             # (130, 7, 9, 0.5, 0.5),
                             (530, 5, 16, 0.1, 0.5),
                             (1300, 10, 2, 0.5, 0.5),
                             (1457, 11, 23, 0.1, 0.5),
                         ])
def test_refinement(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    partitions = random_partition_init(graph.n_nodes, n_partitions)

    reg = _init_szemeredi(n_nodes, n_partitions)
    reg.adj_mat = graph.adjacency
    _create_mock_partition(reg, partitions)
    reg.classes = reg.classes.astype(np.int64)

    reg.certs_compls_list = []
    reg.regularity_list = []
    reg.is_weighted = False
    reg.conditions = [alon1, alon3, alon2]

    n_irr_expected = SzemerediRegularityLemma.check_pairs_regularity(reg)

    np.random.seed(0)
    res = indeg_guided(reg)

    # Format the partition to match the current partition format
    expected_partitions = _reverse_partitions(reg.classes)

    tmp = check_regularity_pairs(graph.adjacency,
                                 n_partitions,
                                 partitions,
                                 epsilon=0.285)
    n_irregular_pairs, certificates_complements, regular_partitions, sze_idx = tmp

    # print(sze_idx, reg.sze_idx)

    assert n_irregular_pairs == n_irr_expected

    # Reset the seed to be sure to have the same results as the other lib
    np.random.seed(0)
    ref = Refinement(graph.adjacency,
                     n_partitions,
                     partitions,
                     certificates_complements,
                     reg.epsilon)

    # Check if the refinement worked
    assert ref.is_refined == res

    # Check if the new partitions equal what is expected from the other lib
    for key, arr in expected_partitions:
        assert np.array_equal(sorted(ref.new_partitions[key]),
                              arr)


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (530, 5, 2, 0.1, 0.5),
                             (530, 5, 4, 0.1, 0.5),
                             (1300, 10, 2, 0.5, 0.5),
                             (1457, 11, 2, 0.1, 0.5),
                         ])
def test_graph_regularity(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    # Create the artificial graph
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    # Create the initial partition
    np.random.seed(0)
    partitions = random_partition_init(n_nodes, n_partitions)

    # print()
    # np.random.seed(0)
    szemeredi_builder = generate_szemeredi_reg_lemma_implementation(kind='alon',
                                                                    sim_mat=graph.adjacency,
                                                                    epsilon=0.285,
                                                                    is_weighted=False,
                                                                    random_initialization=True,
                                                                    refinement='indeg_guided',
                                                                    drop_edges_between_irregular_pairs=True

                                                                    )

    reg = _init_szemeredi(n_nodes, n_partitions)
    _create_mock_partition(reg, partitions)
    # np.random.seed(0)
    import time

    # start_time = time.time()
    result_exp = szemeredi_builder.run(reg.classes,
                                       b=n_partitions,
                                       compression_rate=0.5,
                                       iteration_by_iteration=False,
                                       verbose=False
                                       )
    # print(f'Time old: {time.time() - start_time}')

    is_reg_exp, n_partitions_exp, partitions_exp, sze_idx_exp, regularity_list_exp, n_irreg_pairs_exp = result_exp

    # print(
    #     '######################################################################################################################################')
    # print(
    #     '######################################################################################################################################')
    # print(
    #     '######################################################################################################################################')
    # print(
    #     '######################################################################################################################################')
    # print(
    #     '######################################################################################################################################')

    #######
    np.random.seed(0)
    start_time = time.time()
    results_opt = regularity(graph, n_partitions, epsilon=0.285, compression_rate=0.5, verbose=False)

    # print(f'Time new: {time.time() - start_time}')
    #
    is_reg_opt, n_partitions_opt, partitions_opt, sze_idx_opt, regularity_list_opt, n_irreg_pairs_opt = results_opt

    # print(partitions_opt)
    # # print(partitions_exp)
    # print(_reverse_partitions(partitions_exp))
    # for part_opt, part_exp in zip(partitions_opt, _reverse_partitions(partitions_exp)):
    #     print(part_opt)
    #     print(np.array(part_exp[1]))
    #     assert np.array_equal(part_opt, np.array(part_exp[1]))
    #     pass
    assert n_partitions_opt == n_partitions_exp
    assert all(np.array_equal(part_opt, np.array(part_exp[1])) for part_opt, part_exp in
               zip(partitions_opt, _reverse_partitions(partitions_exp)))
    assert sze_idx_opt == sze_idx_exp
    assert regularity_list_opt == regularity_list_exp
    # Check if all the sub-elements of the regularity list are equal
    assert all(el1 == el2 for el1, el2 in zip(regularity_list_opt, regularity_list_exp))
    assert n_irreg_pairs_opt == n_irreg_pairs_exp


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (530, 5, 2, 0.1, 0.5),
                             (530, 5, 4, 0.1, 0.5),
                             (1300, 10, 2, 0.5, 0.5),
                             (1457, 11, 2, 0.1, 0.5),
                         ])
def test_graph_reduction(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    # Create the artificial graph
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    # Create the initial partition
    np.random.seed(0)
    partitions = random_partition_init(n_nodes, n_partitions)

    print()
    # ##### Expected Behavior ###### #

    reg = _init_szemeredi(n_nodes, n_partitions)
    _create_mock_partition(reg, partitions)

    codec = Codec(0.285, 0.285, 1)
    szemeredi_builder = generate_szemeredi_reg_lemma_implementation(kind='alon',
                                                                    sim_mat=graph.adjacency,
                                                                    epsilon=0.285,
                                                                    is_weighted=False,
                                                                    random_initialization=True,
                                                                    refinement='indeg_guided',
                                                                    drop_edges_between_irregular_pairs=True
                                                                    )

    result_exp = szemeredi_builder.run(reg.classes,
                                       b=n_partitions,
                                       compression_rate=0.5,
                                       iteration_by_iteration=False,
                                       verbose=False
                                       )

    is_reg_exp, n_partitions_exp, partitions_exp, sze_idx_exp, regularity_list_exp, n_irreg_pairs_exp = result_exp
    reduced_mat_exp = codec.reduced_matrix(graph.adjacency,
                                           n_partitions_exp,
                                           0.285,
                                           partitions_exp,
                                           regularity_list_exp)
    # reduced_mat_exp = reduced_mat_exp.astype(get_dtype_adj())

    # ##### Code to test ###### #
    np.random.seed(0)
    results_opt = regularity(graph, n_partitions, epsilon=0.285, compression_rate=0.05, verbose=False)

    is_reg_opt, n_partitions_opt, partitions_opt, sze_idx_opt, regularity_list_opt, n_irreg_pairs_opt = results_opt

    reduced_mat_opt = matrix_reduction(graph.adjacency,
                                       n_partitions_opt,
                                       partitions_opt)

    # ##### Assert section ######
    assert reduced_mat_opt.shape == reduced_mat_exp.shape
    # assert np.allclose(reduced_mat_opt, reduced_mat_exp)
    assert np.array_equal(reduced_mat_opt, reduced_mat_exp)
