from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from rgr.algorithms.conditions.regularity_conditions import RegularityConditions
from rgr.algorithms.partition_pair import PartitionPair, PartitionPairSlow
from rgr.algorithms.refinement import Refinement
from rgr.algorithms.regularity import random_partition_init, check_regularity_pairs, is_partitioning_regular, regularity
from rgr.algorithms.matrix_reduction import matrix_reduction
from rgr.collection.standard import stochastic_block_model
from rgr.constants.types import get_dtype_idx, get_dtype_adj


def main(graph, n_partitions):
    np.random.seed(0)
    results_opt = regularity(graph, n_partitions, epsilon=0.285, compression_rate=0.05, verbose=False)
    # print(results_opt)


def profiling_refinement(graph, n_partitions, parts, eps=0.285):
    np.random.seed(0)



if __name__ == '__main__':
    import cProfile
    np.random.seed(0)
    n_nodes, n_blocks, n_partitions, intra_noise, inter_noise = 5000, 20, 2, 0.2, 0.5
    print(f'Create graph [n={n_nodes}]...')
    graph = stochastic_block_model(n_nodes, n_blocks, intra_noise, inter_noise)
    print(f'Graph created!')

    eps = 0.285
    parts = random_partition_init(graph.n_nodes, n_partitions)

    tmp = check_regularity_pairs(graph.adjacency,
                                 n_partitions,
                                 parts,
                                 epsilon=eps)
    n_irregular_pairs, certificates_complements, regular_partitions, sze_idx = tmp
    cProfile.run('Refinement(graph.adjacency, n_partitions, partitions=parts, certificates_complements=certificates_complements, epsilon=eps)',
                 sort='cumtime')
    # cProfile.run('main(graph, n_partitions)', sort='cumtime')
    # main(graph, n_partitions)
    print(f'End')
