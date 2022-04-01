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
                                   list partitions,
                                   double epsilon):
    """
    
    Args:
        adjacency: 
        partitions: 
        epsilon: 

    Returns:

    """
    cdef:
        int r, s
        int n_partitions, n_irregular_pairs
        double sze_idx
        list certificates_complements, regular_partitions
        CertificatesComplements certs_compls
        PartitionPair pair

    n_partitions = len(partitions)
    n_irregular_pairs, sze_idx = 0, 0.
    certificates_complements = []
    regular_partitions = []

    for r in range(2, n_partitions):
        certificates_complements.append([])
        regular_partitions.append([])

        for s in range(1, r):
            pair = PartitionPair(adjacency, partitions, r, s, eps=epsilon)

            reg_cond = RegularityConditions(pair)
            is_cond_verified, certs_compls = reg_cond.conditions()

            certificates_complements[r - 2].append(certs_compls)

            if is_cond_verified and certs_compls.is_r_certificate_defined():
                n_irregular_pairs += 1
            elif is_cond_verified:
                regular_partitions[r - 2].append(s)

            sze_idx += pair.bip_density ** 2

    sze_idx *= (1.0 / n_partitions ** 2)

    return n_irregular_pairs, certificates_complements, regular_partitions

cpdef bint is_partitioning_regular(int n_irregular_pairs, int n_partitions, double epsilon):
    """
    
    Args:
        n_irregular_pairs: 
        n_partitions: 
        epsilon: 

    Returns:

    """
    cdef double threshold = epsilon * ((n_partitions * (n_partitions - 1)) / 2.)

    return n_irregular_pairs <= threshold

cpdef void regularity(Graph graph, int n_partitions, double epsilon):
    cdef:
        int n_irregular_pairs
        list partitions
        list certificates_complements, regular_partitions

    partitions = random_partition_init(graph.n_nodes, n_partitions)

    while True:
        tmp = check_regularity_pairs(graph.adjacency,
                                     partitions,
                                     epsilon)
        n_irregular_pairs, certificates_complements, regular_partitions = tmp

        if is_partitioning_regular(n_irregular_pairs,
                                   n_partitions,
                                   epsilon):
            print('regular')
            break

        # check if max compression

        Refinement(graph.adjacency,
                   partitions,
                   certificates_complements,
                   epsilon)
