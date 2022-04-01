import numpy as np
cimport numpy as np
from rgr.algorithms.certificates_complements import CertificatesComplements

np.import_array()

cdef class Refinement:

    def __init__(self,
                 np.ndarray[DTYPE_ADJ_t, ndim=2] adjacency,
                 list partitions,
                 list certificates_complements,
                 double epsilon,
                 double threshold=0.5):
        """

        Args:
            adjacency:
            partitions:
            certificates_complements:
            epsilon:
            threshold:
        """
        self.adjacency = adjacency
        self.partitions = partitions
        self.certificates_complements = certificates_complements
        self.epsilon = epsilon
        self.threshold = threshold

        # TODO: solve the use of np.shape
        self.n_nodes = np.array(adjacency).shape[0]
        self.n_partitions = len(partitions)
        self.partition_size = len(partitions[-1])

        self.pair_densities = self._pairwise_densities()

        self.partition_idx = 0
        self.new_partitions = {0: []}
        self._refinement_degree_based()

    cpdef double _density(self,
                          np.ndarray adjacency,
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

        if n_nodes_a == n_nodes_b == 0:
            return 0.

        max_num_edges = n_nodes_a * n_nodes_b
        n_edges = np.sum(adjacency[np.ix_(indices_a, indices_b)])

        return n_edges / max_num_edges

    cpdef np.ndarray _pairwise_densities(self):
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

        densities = np.array([self._density(self.adjacency, partition, partition)
                              for partition in self.partitions])

        return densities

    cpdef bint _refinement_degree_based(self):
        """
        
        Returns:

        """
        cdef:
            CertificatesComplements cur_certs_compls

        to_be_refined = list(range(1, self.n_partitions))

        old_cardinality = self.partition_size
        self.partition_size //= 2

        while to_be_refined:
            s = to_be_refined.pop(0)
            irregular_r_indices = [r for r in to_be_refined
                                   if self.certificates_complements[r - 2][s - 1].is_r_certificate_defined()]

            # If class s has irregular classes
            if irregular_r_indices:
                r = self._choose_candidate(s, irregular_r_indices)
                to_be_refined.remove(r)

                cur_certs_compls = self.certificates_complements[r - 2][s - 1]

                # Merging the two complements
                compls = np.append(cur_certs_compls.s_compls, cur_certs_compls.r_compls)

                set1_s, set2_s, compls = self._update_certs(cur_certs_compls.s_certs, compls)
                set1_r, set2_r, compls = self._update_certs(cur_certs_compls.r_certs, compls)

                self._update_partitions(set1_s, set2_s)
                self._update_partitions(set1_r, set2_r)

                if compls.size > 0:
                    self.new_partitions[0].extend(compls)
            else:
                # The class is e-reg with all the others or it does not have irregular classes

                # Sort by indegree and unzip the structure
                s_indices = self.partitions[s]
                s_indegs = np.sum(self.adjacency[np.ix_(s_indices, s_indices)], axis=1).argsort()

                set1 = s_indices[s_indegs[0:][::2]]
                set2 = s_indices[s_indegs[1:][::2]]

                self._update_partitions(set1, set2)

        self.n_partitions *= 2

        self.new_partitions[0].extend(self.partitions[0])
        self.new_partitions[0] = np.array(self.new_partitions[0], dtype=DTYPE_IDX)

        if self.new_partitions[0].size >= (self.epsilon * self.n_nodes):
            if self.new_partitions[0].size > self.n_partitions:
                pass
                # TODO: handle the C_0 > size_partition
                # self.new_partitions[0][:self.n_partitions]
            else:
                return False

        # TODO: modify the dict to list of partitions
        # partitions *= -1

        return True

    cpdef int _choose_candidate(self, int s, list irregulars):
        """
        
        Args:
            s: int
            irregulars: list

        Returns:

        """
        cdef:
            int candidate = -1
            double candidate_dens, s_dens, r_dens

        candidate_dens = float('-inf')

        # Exploit the precalculated densities
        s_dens = self.pair_densities[s]
        for r in irregulars:
            r_dens = self._density_candidate(self.partitions[s],
                                             self.partitions[r],
                                             s_dens,
                                             r)
            if r_dens > candidate_dens:
                candidate_dens = r_dens
                candidate = r

        return candidate

    cdef double _density_candidate(self,
                                   np.ndarray s_indices,
                                   np.ndarray r_indices,
                                   double s_dens,
                                   int r):
        return self._density(self.adjacency, s_indices, r_indices) + (1 - abs(s_dens - self.pair_densities[r]))

    cpdef tuple _update_certs(self, np.ndarray cert, np.ndarray compls):
        """
        
        Args:
            cert: np.ndarray
            compls: np.ndarray
            dens_cert: double

        Returns:

        """
        cdef:
            double dens_cert

        dens_cert = self._density(self.adjacency, cert, cert)
        degrees = np.sum(self.adjacency[np.ix_(cert, cert)], axis=1).argsort()[::-1]

        if dens_cert > self.threshold:
            set1 = cert[degrees[0::2]]
            set2 = cert[degrees[1::2]]
            maximize_density = True
        else:
            set1 = np.random.choice(cert, cert.size // 2, replace=False)
            set2 = np.setdiff1d(cert, set1)
            maximize_density = False

        set1, compls = self._fill_new_set(set1, compls, maximize_density)
        set2, compls = self._fill_new_set(set2, compls, maximize_density)

        return set1, set2, compls

    cpdef tuple _fill_new_set(self, new_set, compls, maximize_density):
        """ Find nodes that can be added
        Move from compls the nodes in can_be_added until we either finish the nodes or reach the desired cardinality
        :param new_set: np.array(), array of indices of the set that must be augmented
        :param compls: np.array(), array of indices used to augment the new_set
        :param maximize_density: bool, used to augment or decrement density
        """

        val = 1 if maximize_density else 0
        nodes = self.adjacency[np.ix_(new_set, compls)] == val

        to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        to_add = to_add[0][to_add[1].argsort()]

        if maximize_density:
            to_add = to_add[::-1]

        while new_set.size < self.partition_size:

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

    cpdef void _update_partitions(self, np.ndarray part1, np.ndarray part2):
        self._update_partition(part1)
        self._update_partition(part2)

    cpdef void _update_partition(self, np.ndarray partition):
        self.partition_idx -= 1
        self.new_partitions[self.partition_idx] = partition

        if partition.size > self.partition_size:
            last, self.new_partitions[self.partition_idx] = self.new_partitions[self.partition_idx][-1], \
                                                            self.new_partitions[self.partition_idx][::-1]
            self.new_partitions[0].append(last)
