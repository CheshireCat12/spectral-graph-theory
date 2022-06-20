import numpy as np

from rgr.algorithms.conditions.regularity_conditions import RegularityConditions
from rgr.algorithms.partition_pair import PartitionPair
from rgr.algorithms.refinement import Refinement

cpdef list random_partition_init(int n_nodes, int n_partitions, bint sort_partitions=True):
    """
    Create the initial random partitioning of Alon algorithm.
    The nodes in graph G are partitioned into ``n_partitions`` partitions.
    Where each class C_{``n_partitions``} has an equal number of elements
    
    Args:
        n_nodes: int
            Number of nodes
        n_partitions: int
            Number of partitions
        sort_partitions: bool
            If ``True`` sort the indices in all partitions

    Returns:
        List - List of np.ndarray containing the partitions.
               Each element of the list contains the idx of the given partition.
    """
    cdef:
        int prt_size
        list partitions
        np.ndarray indices, idx_split

    prt_size = n_nodes // n_partitions

    indices = np.random.permutation(n_nodes).astype(dtype=DTYPE_IDX)

    # Split the shuffled indices into the partitions
    idx_split = np.arange((n_nodes - (n_partitions * prt_size)),
                          (n_nodes - prt_size + 1),
                          prt_size)
    partitions = np.split(indices, idx_split)

    if sort_partitions:
        partitions = [np.sort(partition) for partition in partitions]

    return partitions

cpdef tuple check_regularity_pairs(np.ndarray adjacency,
                                   int n_partitions,
                                   list partitions,
                                   double epsilon):
    """
    Check if the pairwise partitions are eps-regular.
    
    Args:
        adjacency: np.ndarray
            Adjacency matrix of the graph
        n_partitions: int
            Number of partitions
        Partitions: List
            List of the partitions
        epsilon: float
            Epsilon parameter

    Returns:

    """
    cdef:
        int r, s
        int n_irregular_pairs
        double sze_idx
        list certificates_complements, regular_partitions
        CertificatesComplements certs_compls
        PartitionPair pair

    n_irregular_pairs, sze_idx = 0, 0.
    certificates_complements = []
    regular_partitions = []

    # print(f'n_partitions {n_partitions + 1}')
    for r in range(2, n_partitions + 1):
        certificates_complements.append([])
        regular_partitions.append([])

        for s in range(1, r):
            pair = PartitionPair(adjacency, partitions, r, s, eps=epsilon)

            reg_cond = RegularityConditions(pair)
            is_cond_verified, certs_compls = reg_cond.conditions()

            certificates_complements[r - 2].append(certs_compls)

            if r == 2 and s == 1:
                # print(certs_compls)
                # print(certs_compls.r_certs)
                pass
            # print(certs_compls.r_certs)
            # print(f'is cond ver {is_cond_verified}')
            if is_cond_verified and certs_compls.is_r_certificate_defined():
                n_irregular_pairs += 1
                # print('increment n_irreg_pairs')
                # print(f'{r}-{s}-increment n_irreg_pairs')
            elif is_cond_verified:
                # print(f'{r}-{s}-regularity list')
                regular_partitions[r - 2].append(s)

            sze_idx += pair.bip_density ** 2.0
            # print(f'bib dens - {pair.bip_density}')
            # print(sze_idx)

    sze_idx *= (1.0 / n_partitions ** 2.0)
    # print(f'final - {sze_idx}')
    # print(f'n_partitions - {n_partitions}')
    return n_irregular_pairs, certificates_complements, regular_partitions, sze_idx

cpdef bint is_partitioning_regular(int n_irregular_pairs, int n_partitions, double epsilon):
    """
    
    Args:
        n_irregular_pairs: 
        n_partitions: 
        epsilon: 

    Returns:

    """
    cdef double threshold = epsilon * ((n_partitions * (n_partitions - 1)) / 2.)

    # print(f'n_partition_threshold {n_partitions}')
    # print(f'num irreg parts {n_irregular_pairs}')
    # print(f'threshold {threshold}')

    return n_irregular_pairs <= threshold

cpdef tuple regularity(Graph graph,
                       int n_partitions,
                       double epsilon,
                       double compression_rate,
                       bint verbose=False):
    cdef:
        int iteration = 0
        int n_irregular_pairs, max_n_partitions
        double sze_idx
        list partitions
        list certificates_complements, regular_partitions
        tuple tmp_elements

    max_n_partitions = int(compression_rate * graph.n_nodes)

    # np.random.seed(0)
    partitions = random_partition_init(graph.n_nodes, n_partitions)

    # np.random.seed(0)
    while True:
        iteration += 1

        tmp_elements = check_regularity_pairs(graph.adjacency,
                                              n_partitions,
                                              partitions,
                                              epsilon=epsilon)
        n_irregular_pairs, certificates_complements, regular_partitions, sze_idx = tmp_elements
        # print(n_irregular_pairs)
        if is_partitioning_regular(n_irregular_pairs,
                                   n_partitions,
                                   epsilon):
            if verbose:
                print(f'{iteration}, {n_partitions}, {partitions[1].size}, {sze_idx}, regular')
            break

        # check if max compression
        # TODO: check if the max compression is reached!

        if verbose:
            print(f'{iteration}, {n_partitions}, {partitions[1].size}, {sze_idx}, irregular')
        refinement_verbose = n_partitions >= 16
        refinement = Refinement(graph.adjacency,
                                n_partitions,
                                partitions,
                                certificates_complements,
                                epsilon,
                                verbose=refinement_verbose)

        # print(refinement.is_refined)
        if not refinement.is_refined:
            print('not regular!')
            break

        n_partitions = refinement.n_partitions
        partitions = refinement.new_partitions
        # print(f'pair densities refinement {refinement.pair_densities}')
        # print('*****************************************')

    # TODO: check if n_partitions corresponds to the k param
    return True, n_partitions, partitions, sze_idx, regular_partitions, n_irregular_pairs
