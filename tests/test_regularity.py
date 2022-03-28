from rgr.algorithms.regularity import random_partition_init
import pytest
import numpy as np
from collections import Counter


@pytest.mark.parametrize('n_nodes, n_partitions, expected',
                         [
                             (30, 5, np.array(
                                 [1, 5, 3, 2, 5, 5, 5, 2, 3, 4, 1, 3, 2, 3, 4, 4, 1, 5, 2, 1, 4, 4, 2, 2, 5, 1, 1, 4, 3,
                                  3])),
                             (33, 5, np.array(
                                 [1, 5, 3, 2, 5, 5, 5, 2, 3, 4, 1, 3, 2, 3, 4, 4, 1, 5, 2, 1, 4, 4, 2, 2, 5, 1, 1, 4, 3,
                                  3])),
                             (163, 15, np.array([12, 7, 4, 11, 1, 11, 5, 9, 8, 8, 14, 6, 8, 6, 7, 4, 8,
                                                 10, 5, 2, 13, 7, 3, 1, 13, 3, 5, 10, 10, 3, 14, 9, 3, 13,
                                                 14, 6, 2, 9, 7, 10, 12, 1, 15, 5, 2, 7, 12, 15, 12, 11, 7,
                                                 14, 6, 9, 13, 14, 11, 15, 6, 15, 9, 4, 11, 10, 7, 9, 10, 13,
                                                 5, 2, 12, 10, 3, 2, 6, 1, 15, 10, 1, 7, 11, 10, 2, 11, 13,
                                                 11, 13, 5, 2, 12, 13, 5, 15, 1, 12, 14, 5, 1, 13, 2, 4, 1,
                                                 6, 15, 11, 1, 4, 3, 6, 8, 4, 3, 8, 4, 12, 6, 14, 7, 4,
                                                 14, 2, 13, 2, 3, 14, 5, 14, 10, 9, 8, 12, 15, 8, 8, 3, 9,
                                                 15, 15, 4, 6, 9, 8, 9, 4, 3, 1, 11, 7, 12, 5],
                                                dtype=np.int32))
                         ])
def test_partition_init(n_nodes, n_partitions, expected):
    np.random.seed(0)
    partition = random_partition_init(n_nodes, n_partitions)

    assert np.array_equal(partition, expected)
    # print()
    # print(19 % 5)
    # print(len(partition))
    # print(partition)
    # print(Counter(partition))
    # print(sum(Counter(partition).values()))
    # classes = np.zeros(n_nodes)
    # classes_cardinality = n_nodes // n_partitions
    #
    # for i in range(n_partitions):
    #     classes[(i * classes_cardinality):((i + 1) * classes_cardinality)] = i + 1
    #
    # # np.random.shuffle(classes)
    # print('final')
    # print(classes)
    # print(Counter(classes))
    # print(sum(Counter(classes).values()))
