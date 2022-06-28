import numpy as np
import pytest

from rgr.algorithms.regularity import random_partition_init

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
