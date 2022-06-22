import timeit

import numpy as np
import pytest

from rgr.algorithms.partition_pair import sum_matrix, sum_mat_axis, PartitionPair, PartitionPairFast

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
           pair = PartitionPair(adj, partitions, r, s, eps=0.285)
    print(f"Time difference is (naive): {(timeit.default_timer() - starttime) * 1000}")

    starttime = timeit.default_timer()
    for r in range(2, n_partitions + 1):
        for s in range(1, r):
            pair = PartitionPairFast(adj, partitions, r, s, eps=0.285)
    print(f"Time difference is (fast): {(timeit.default_timer() - starttime) * 1000}")
