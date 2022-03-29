import numpy as np

cpdef double density(np.ndarray adjacency,
                     np.ndarray indices_a,
                     np.ndarray indices_b):
    """
    Compute the edge density between two set of vertices
    
    Args:
        adjacency: np.ndarray
            Adjacency matrix
        indices_a: np.ndarray
            Indices of the first set
        indices_b: np.ndarray
            Indices of the second set

    Returns: float - Edge density

    """
    cdef:
        int n_nodes_a, n_nodes_b
        int max_num_edges
        int n_edges
    n_nodes_a = indices_a.size
    n_nodes_b = indices_b.size
    max_num_edges = n_nodes_a * n_nodes_b
    n_edges = np.sum(adjacency[np.ix_(indices_a, indices_b)])

    return n_edges / max_num_edges

cpdef np.ndarray pairwise_densities(np.ndarray adjacency, list partitions):
    """
    Compute the pairwise density between the all the partitions
    
    Args:
        adjacency: np.ndarray
            Adjacency matrix
        partitions: list
            Array containing the idx for each partition

    Returns: np.ndarray - Array of intra-partition density

    """
    cdef np.ndarray densities

    densities = np.array([density(adjacency, partition, partition)
                          for partition in partitions])

    return densities

def choose_candidate(adjacency, partitions, in_densities, s, irregulars):
    """ This function chooses a class between the irregular ones (d(ci,cj), 1-|d(ci,ci)-d(cj,cj)|)
    :param in_densities: list(float), precomputed densities to speed up the calculations
    :param s: int, the class which all the other classes are compared to
    :param irregulars: list(int), the list of irregular classes
    """
    candidate_idx = -1
    candidate = -1

    # Exploit the precalculated densities
    s_dens = in_densities[s]
    for r in irregulars:
        s_indices = partitions[s]  # np.where(self.classes == s)[0]
        r_indices = partitions[r]  # np.where(self.classes == r)[0]
        r_idx = density(adjacency, s_indices, r_indices) + (1 - abs(s_dens - in_densities[r]))
        if r_idx > candidate_idx:
            candidate_idx = r_idx
            candidate = r

    return candidate

def fill_new_set(adjacency, prt_size, new_set, compls, maximize_density):
    """ Find nodes that can be added
    Move from compls the nodes in can_be_added until we either finish the nodes or reach the desired cardinality
    :param new_set: np.array(), array of indices of the set that must be augmented
    :param compls: np.array(), array of indices used to augment the new_set
    :param maximize_density: bool, used to augment or decrement density
    """

    if maximize_density:
        nodes = adjacency[np.ix_(new_set, compls)] == 1.0
        #nodes = self.sim_mat[np.ix_(new_set, compls)] >= 0.5

        # These are the nodes that can be added to certs, we take the most connected ones with all the others
        to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        to_add = to_add[0][to_add[1].argsort()]
    else:
        nodes = adjacency[np.ix_(new_set, compls)] == 0.0
        #nodes = self.sim_mat[np.ix_(new_set, compls)] < 0.5

        # These are the nodes that can be added to certs, we take the less connected ones with all the others
        to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        to_add = to_add[0][to_add[1].argsort()[::-1]]

    while new_set.size < prt_size:

        # If there are nodes in to_add, we keep moving from compls to new_set
        if to_add.size > 0:
            node, to_add = to_add[-1], to_add[:-1]
            new_set = np.append(new_set, node)
            compls = np.delete(compls, np.argwhere(compls == node))

        else:
            # If there aren't candidate nodes, we keep moving from complements
            # to certs until we reach the desired cardinality
            node, compls = compls[-1], compls[:-1]
            new_set = np.append(new_set, node)

    return new_set, compls

cpdef list irregular_r_partitions(list certificates_complements, list to_be_refined, int s):
    cdef list irregular_r_indices = []

    for r in to_be_refined:
        if certificates_complements[r - 2][s - 1][0][0]:
            irregular_r_indices.append(r)

    return irregular_r_indices

cpdef refinement_degree_based(adjacency,
                              partitions,
                              certificates_complements,
                              epsilon):
    """ In-degree based refinemet. The refinement exploits the internal structure of the classes of a given partition.
    :returns: True if the new partition is valid, False otherwise
    """
    cdef:
        int n_partitions, partition_size
        double threshold

    threshold = 0.5
    n_partitions = len(partitions)
    partition_size = len(partitions[1])


    to_be_refined = list(range(1, n_partitions))

    old_cardinality = partition_size
    partition_size //= 2
    in_densities = pairwise_densities(adjacency, partitions[1:])
    new_k = 0
    new_partitions = dict()

    while to_be_refined:
        s = to_be_refined.pop(0)
        irregular_r_indices = irregular_r_partitions(certificates_complements,
                                                     to_be_refined,
                                                     s)


        # If class s has irregular classes
        if irregular_r_indices:

            # Choose candidate based on the inside-outside density index
            r = choose_candidate(adjacency,
                                 partitions,
                                 in_densities,
                                 s,
                                 irregular_r_indices)
            to_be_refined.remove(r)

            s_certs = np.array(certificates_complements[r - 2][s - 1][0][1]).astype('int32')
            s_compls = np.array(certificates_complements[r - 2][s - 1][1][1]).astype('int32')
            assert s_certs.size + s_compls.size == old_cardinality

            r_compls = np.array(certificates_complements[r - 2][s - 1][1][0]).astype('int32')
            r_certs = np.array(certificates_complements[r - 2][s - 1][0][0]).astype('int32')
            assert r_certs.size + r_compls.size == old_cardinality

            # Merging the two complements
            compls = np.append(s_compls, r_compls)

            # Calculating certificates densities
            dens_s_cert = density(adjacency, s_certs, s_certs)
            dens_r_cert = density(adjacency, r_certs, r_certs)

            for cert, dens in [(s_certs, dens_s_cert), (r_certs, dens_r_cert)]:

                # Indices of the cert ordered by in-degree, it doesn't matter if we reverse the list as long as we unzip it
                degs = adjacency[np.ix_(cert, cert)].sum(1).argsort()[::-1]
                #degs = self.sim_mat[np.ix_(cert, cert)].sum(1).argsort()[::-1]

                if dens > threshold:
                    # Certificates high density branch

                    # Unzip them in half to preserve seeds
                    set1 = cert[degs[0:][::2]]
                    set2 = cert[degs[1:][::2]]

                    # Adjust cardinality of the new set to the desired cardinality
                    set1, compls = fill_new_set(adjacency, prt_size, set1, compls, True)
                    set2, compls = fill_new_set(adjacency, prt_size, set2, compls, True)

                    # # Handling of odd classes
                    # new_k -= 1
                    # self.classes[set1] = new_k
                    # if set1.size > self.classes_cardinality:
                    #     self.classes[set1[-1]] = 0
                    # new_k -= 1
                    # self.classes[set2] = new_k
                    # if set2.size > self.classes_cardinality:
                    #     self.classes[set2[-1]] = 0

                else:
                    # Certificates low density branch
                    set1 = np.random.choice(cert, len(cert) // 2, replace=False)
                    set2 = np.setdiff1d(cert, set1)

                    # Adjust cardinality of the new set to the desired cardinality
                    set1, compls = fill_new_set(adjacency, prt_size, set1, compls, False)
                    set2, compls = fill_new_set(adjacency, prt_size, set2, compls, False)

                # Handling of odd classes
                new_k -= 1
                partitions[set1] = new_k
                if set1.size > prt_size:
                    partitions[set1[-1]] = 0

                new_k -= 1
                partitions[set2] = new_k
                if set1.size > prt_size:
                    partitions[set1[-1]] = 0
                # # Handling of odd classes
                # new_k -= 1
                # self.classes[set1] = new_k
                # if set1.size > self.classes_cardinality:
                #     self.classes[set1[-1]] = 0

                # new_k -= 1
                # self.classes[set2] = new_k
                # if set2.size > self.classes_cardinality:
                #     self.classes[set2[-1]] = 0

                # # Handle special case when there are still some complements not assigned
                # if compls.size > 0:
                #     self.classes[compls] = 0

        else:
            # The class is e-reg with all the others or it does not have irregular classes

            # Sort by indegree and unzip the structure
            s_indices = partitions[s]
            s_indegs = np.sum(adjacency[np.ix_(s_indices, s_indices)], axis=1).argsort()

            set1 = s_indices[s_indegs[0:][::2]]
            set2 = s_indices[s_indegs[1:][::2]]

            # Handling of odd classes
            new_k -= 1
            new_partitions[new_k] = set1

            if set1.size > partition_size:
                last, new_partitions[new_k] = new_partitions[new_k][-1], new_partitions[new_k][::-1]
                new_partitions.get(0, []).append(last)

            new_k -= 1
            new_partitions[new_k] = set2
            if set2.size > partition_size:
                last, new_partitions[new_k] = new_partitions[new_k][-1], new_partitions[new_k][::-1]
                new_partitions.get(0, []).append(last)

    n_partitions *= 2

    # Check validity of class C0, if invalid and enough nodes, distribute the exceeding nodes among the classes
    c0_indices = partitions[0]
    if c0_indices.size >= (epsilon * adjacency.shape[0]):
        if c0_indices.size > prt_size:
            partitions[c0_indices[:n_partitions]] = np.array(range(1, n_partitions + 1)) * -1
        else:
            print('[ refinement ] Invalid cardinality of C_0')
            return False

    partitions *= -1

    # if not partition_correct(self):
    #     ipdb.set_trace()
    return True
