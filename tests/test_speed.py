import timeit

import numpy as np
import pytest

from rgr.algorithms.partition_pair import sum_matrix, sum_mat_axis, PartitionPair, PartitionPairSlow

from rgr.algorithms.regularity import random_partition_init, check_regularity_pairs, is_partitioning_regular, regularity


def test_speed_compute():
    print()

    arr = np.ones((10000, 10000), dtype=np.int8)

    starttime = timeit.default_timer()
    exp_val = np.sum(arr)
    print(f"Time difference is (NP): {(timeit.default_timer() - starttime) * 1000}")

    starttime = timeit.default_timer()
    opt_val = sum_matrix(arr)
    print(f"Time difference is (Cython): {(timeit.default_timer() - starttime) * 1000}")

    assert opt_val == exp_val


@pytest.mark.parametrize('size, axis',
                         [
                             ((1000, 400), 0),
                             ((1000, 400), 1),
                         ])
def test_speed_sum_mat_axis(size, axis):
    print()
    np.random.seed(0)
    arr = np.random.random_integers(0, 1, size=size).astype(np.int8)
    # print(arr)
    starttime = timeit.default_timer()
    exp_val = np.sum(arr, axis=axis)
    print(f"Time difference is (NP): {(timeit.default_timer() - starttime) * 1000}")

    starttime = timeit.default_timer()
    opt_val = sum_mat_axis(arr, axis=axis)
    print(f"Time difference is (Cython): {(timeit.default_timer() - starttime) * 1000}")
    # print(exp_val)
    # print(np.array(opt_val))
    assert np.array_equal(np.array(opt_val), exp_val)


@pytest.mark.parametrize('n_nodes, n_partitions',
                         [
                             (500, 5),
                             (10000, 5),
                             (10000, 50),
                             (10000, 500),
                             # (10000, 5000),
                         ])
def test_speed_partition_pair(n_nodes, n_partitions):
    print()
    np.random.seed(0)
    adj = np.random.random_integers(0, 1, size=(n_nodes, n_nodes)).astype(np.int8)

    partitions = random_partition_init(n_nodes, n_partitions)

    # starttime = timeit.default_timer()
    # PartitionPair(adj, partitions, 2, 1, 0.285)
    # print(f"Time difference is (naive): {(timeit.default_timer() - starttime) * 1000}")
    #
    # starttime = timeit.default_timer()
    # PartitionPairFast(adj, partitions, 2, 1, 0.285)
    # print(f"Time difference is (fast): {(timeit.default_timer() - starttime) * 1000}")

    starttime = timeit.default_timer()
    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPairSlow(adj, partitions, r, s, eps=0.285)
    print(f"Time difference is (naive): {(timeit.default_timer() - starttime) * 1000}")

    starttime = timeit.default_timer()
    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPair(adj, partitions, r, s, eps=0.285)
    print(f"Time difference is (fast): {(timeit.default_timer() - starttime) * 1000}")


from rgr.constants.types import get_dtype_idx, get_dtype_adj

from dataclasses import dataclass

from rgr.algorithms.conditions.regularity_conditions import RegularityConditions
from rgr.algorithms.partition_pair import PartitionPair, PartitionPairSlow
from rgr.algorithms.regularity import random_partition_init, check_regularity_pairs, is_partitioning_regular, regularity
from rgr.collection.standard import stochastic_block_model
from rgr.constants.types import get_dtype_idx, get_dtype_adj, get_dtype_uint

from typing import List


@dataclass
class MockSzemeredi:
    N: int
    k: int
    epsilon: float
    classes: np.ndarray
    classes_cardinality: int


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


from rgr.algorithms.conditions.regularity_conditions import find_Yp, find_Yp_fast, neighbourhood_deviation, \
    neighbourhood_deviation_fast, compute_y0_fast, compute_y0


@pytest.mark.parametrize('n_nodes, n_blocks, n_partitions, intra_noise, inter_noise',
                         [
                             (530, 5, 2, 0.1, 0.5),
                             # (5300, 2, 2, 0.1, 0.5),
                         ])
def test_speed_conditions(n_nodes, n_blocks, n_partitions, intra_noise, inter_noise):
    np.random.seed(0)
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    partitions = random_partition_init(n_nodes, n_partitions)

    reg = _init_szemeredi(n_nodes, n_partitions)
    reg.adj_mat = graph.adjacency
    _create_mock_partition(reg, partitions)
    print()

    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPairSlow(graph.adjacency, partitions, r, s, eps=0.285)
            pair_f = PartitionPair(graph.adjacency, partitions, r, s, eps=0.285)

            reg_cond = RegularityConditions(pair)

            conditions = [reg_cond.condition_1, reg_cond.condition_2, reg_cond.condition_3]

            starttime = timeit.default_timer()
            mat = neighbourhood_deviation(pair.bip_adj, pair.bip_avg_deg, pair.prts_size)
            print(f"neighbor def - Time difference is (naive): {(timeit.default_timer() - starttime) * 1000}")

            starttime = timeit.default_timer()
            mat_f = neighbourhood_deviation_fast(pair_f.bip_adj, pair_f.bip_avg_deg, pair.prts_size)
            print(f"neighbor def - Time difference is (fast): {(timeit.default_timer() - starttime) * 1000}")

            starttime = timeit.default_timer()
            yp_i = find_Yp(pair.s_degrees, pair.bip_avg_deg, reg_cond.threshold_dev)
            print(f"Yp - Time difference is (naive): {(timeit.default_timer() - starttime) * 1000}")

            starttime = timeit.default_timer()
            yp_i_f = find_Yp_fast(pair_f.s_degrees, pair_f.bip_avg_deg, reg_cond.threshold_dev)
            print(f"Yp - Time difference is (fast): {(timeit.default_timer() - starttime) * 1000}")

            assert np.array_equal(np.array(yp_i_f), yp_i)
            starttime = timeit.default_timer()
            s_certs, y0 = compute_y0(mat,
                                     pair.s_indices,
                                     yp_i.astype(get_dtype_uint()),
                                     reg_cond.threshold_dev)
            print(f"Yo - Time difference is (naive): {(timeit.default_timer() - starttime) * 1000}")

            starttime = timeit.default_timer()
            s_certs_f, y0_f = compute_y0_fast(mat,
                                     pair.s_indices,
                                     yp_i.astype(get_dtype_uint()),
                                     reg_cond.threshold_dev)
            print(f"Yo - Time difference is (fast): {(timeit.default_timer() - starttime) * 1000}")

            assert y0_f == y0
            assert np.array_equal(np.array(s_certs_f), s_certs)
            # s_certs, y0 = compute_y0(ngh_dev,
            #                          self.pair.s_indices,
            #                          yp_filter,
            #                          self.threshold_dev)

            # for condition in conditions:
            #     verified, certs_compls = condition()

            break
        break
