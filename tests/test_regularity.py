from rgr.algorithms.regularity import random_partition_init

import numpy as np
from collections import Counter
def test_partition_init():
    np.random.seed(0)
    n_nodes = 19
    n_partitions = 5
    partition = random_partition_init(19, 5)
    print()
    print(19 % 5)
    print(len(partition))
    print(partition)
    print(Counter(partition))
    print(sum(Counter(partition).values()))
    classes = np.zeros(n_nodes)
    classes_cardinality = n_nodes // n_partitions

    for i in range(n_partitions):
        classes[(i * classes_cardinality):((i + 1) * classes_cardinality)] = i + 1

    np.random.shuffle(classes)
    print('final')
    print(classes)
    print(Counter(classes))
    print(sum(Counter(classes).values()))
