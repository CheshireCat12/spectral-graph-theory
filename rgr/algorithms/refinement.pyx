import numpy as np
cimport numpy as np
cimport cython
from rgr.algorithms.certificates_complements import CertificatesComplements

np.import_array()

cdef class Refinement:


    @cython.profile(True)
    def __init__(self,
                 np.ndarray[DTYPE_ADJ_t, ndim=2, mode='c'] adjacency,
                 int n_partitions,
                 list partitions,
                 list certificates_complements,
                 double epsilon,
                 double threshold=0.5,
                 bint verbose=False):
        """

        Args:
            adjacency:
            partitions:
            certificates_complements:
            epsilon:
            threshold:
        """
        self.adjacency = adjacency
        self.n_partitions = n_partitions
        self.partitions = partitions
        self.certificates_complements = certificates_complements
        self.epsilon = epsilon
        self.threshold = threshold
        self.verbose = verbose

        self.n_nodes = np.array(adjacency).shape[0]
        self.partition_size = len(partitions[-1])

        self.pair_densities = self._pairwise_densities()

        self.partition_idx = 0
        self.new_dict_partitions = {0: []}
        self.is_refined = self._refinement_degree_based()

        # TODO: modify the dict to list of partitions
        self.new_partitions = [np.sort(val) for key, val in
                               sorted(self.new_dict_partitions.items(),
                                      key=lambda x: -1 * x[0])]
        # if self.verbose:
        #     print([list(val) for val in self.new_partitions])

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double _density(self,
                          DTYPE_ADJ_t[:, ::1] adjacency,
                          DTYPE_UINT_t[::1] indices_a,
                          DTYPE_UINT_t[::1] indices_b,
                          bint same_index_set=False):
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
            Py_ssize_t i, j
            bint size_is_zero, size_is_one
            int n_nodes_a, n_nodes_b
            int idx_a, idx_b
            int max_num_edges
            int n_edges = 0
        n_nodes_a = indices_a.shape[0]
        n_nodes_b = indices_b.shape[0]

        size_is_zero = n_nodes_a == n_nodes_b == 0
        size_is_one = n_nodes_a == n_nodes_b == 1
        if size_is_zero or size_is_one:
            return 0.

        if same_index_set:
            max_num_edges = (n_nodes_a * (n_nodes_a - 1)) // 2
            # n_edges = np.sum(np.tril(np.asarray(adjacency)[np.ix_(np.asarray(indices_a), np.asarray(indices_b))], -1))
            # print(f'n_edges old {n_edges}')
            for i in range(1, n_nodes_a):
                idx_a = indices_a[i]
                for j in range(i):
                    idx_b = indices_b[j]
                    n_edges += adjacency[idx_a][idx_b]

            # print(f'n_edges new {n_edges}')
        else:
            max_num_edges = n_nodes_a * n_nodes_b

            for i in range(n_nodes_a):
                idx_a = indices_a[i]
                for j in range(n_nodes_b):
                    idx_b = indices_b[j]
                    # print(adjacency[idx_a][idx_b])
                    n_edges += adjacency[idx_a][idx_b]
            # n_edges = np.sum(adjacency[np.ix_(indices_a, indices_b)])
            # print(f'old {n_edges}')

        return n_edges / max_num_edges

    @cython.profile(True)
    cpdef np.ndarray _pairwise_densities(self):
        """
        Compute the pairwise density between all partitions
    
        Args:
            adjacency: np.ndarray
                Adjacency matrix
            partitions: list
                Array containing the idx for each partition
    
        Returns: np.ndarray - Array of intra-partition density
    
        """
        cdef np.ndarray densities

        densities = np.array([self._density(self.adjacency, partition, partition, same_index_set=True)
                              for partition in self.partitions], dtype='float32')

        return densities

    @cython.profile(True)
    cpdef bint _refinement_degree_based(self):
        """
        
        Returns:

        """
        cdef:
            int r, s
            int old_cardinality
            list to_be_refined, irregular_r_indices
            np.ndarray compls
            CertificatesComplements cur_certs_compls

        to_be_refined = list(range(1, self.n_partitions + 1))

        old_cardinality = self.partition_size
        self.partition_size //= 2

        while to_be_refined:
            # print(f'to be refined {to_be_refined}')
            s = to_be_refined.pop(0)
            irregular_r_indices = [r for r in to_be_refined
                                   if self.certificates_complements[r - 2][s - 1].is_r_certificate_defined()]
            # print(irregular_r_indices)
            # print(f'irreg r idx: {irregular_r_indices}')
            # If class s has irregular classes
            if irregular_r_indices:
                r = self._choose_candidate(s, irregular_r_indices)
                # print(r)
                to_be_refined.remove(r)

                cur_certs_compls = self.certificates_complements[r - 2][s - 1]

                # Merging the two complements
                compls = np.append(cur_certs_compls.s_compls, cur_certs_compls.r_compls)

                # if self.verbose:
                #     print(f'certificat compls', compls)

                set1_s, set2_s, compls = self._update_certs(cur_certs_compls.s_certs, compls)
                set1_r, set2_r, compls = self._update_certs(cur_certs_compls.r_certs, compls)

                self._update_partitions(set1_s, set2_s)
                self._update_partitions(set1_r, set2_r)

                if compls.size > 0:
                    self.new_dict_partitions[0].extend(compls)
            else:
                # The class is e-reg with all the others or it does not have irregular classes

                # if self.verbose:
                #     print(f'e-reg')
                # Sort by indegree and unzip the structure
                s_indices = self.partitions[s]
                s_indegs = np.sum(self.adjacency[np.ix_(s_indices, s_indices)], axis=1).argsort()

                set1 = s_indices[s_indegs[0:][::2]]
                set2 = s_indices[s_indegs[1:][::2]]

                self._update_partitions(set1, set2)

                # print('after sets')
                # print(set1)
                # print(set2)
            # print('partitions')
            # print(self.new_partitions)

        self.n_partitions *= 2

        self.new_dict_partitions[0].extend(self.partitions[0])
        self.new_dict_partitions[0] = np.array(self.new_dict_partitions[0], dtype=DTYPE_IDX)

        if self.new_dict_partitions[0].size >= (self.epsilon * self.n_nodes):
            if self.new_dict_partitions[0].size > self.n_partitions:
                pass
                # TODO: handle the C_0 > size_partition
                # self.new_partitions[0][:self.n_partitions]
                assert False, 'wrong refinement.pyx line 174'
            else:
                return False

        return True

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef int _choose_candidate(self, int s, list irregulars):
        """
        
        Args:
            s: int
            irregulars: list

        Returns:

        """
        cdef:
            Py_ssize_t i, max_i
            int candidate = -1
            int r
            double candidate_dens, s_dens, r_dens

        candidate_dens = float('-inf')

        # Exploit the precalculated densities
        s_dens = self.pair_densities[s]

        max_i = len(irregulars)
        for i in range(max_i):
            r = irregulars[i]

            r_dens = self._density_candidate(self.partitions[s],
                                             self.partitions[r],
                                             s_dens,
                                             r)
            if r_dens > candidate_dens:
                candidate_dens = r_dens
                candidate = r

        return candidate

    @cython.profile(True)
    cdef double _density_candidate(self,
                                   np.ndarray s_indices,
                                   np.ndarray r_indices,
                                   double s_dens,
                                   int r):
        return self._density(self.adjacency, s_indices, r_indices) + (1 - abs(s_dens - self.pair_densities[r]))

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef tuple _update_certs(self, np.ndarray cert, np.ndarray compls):
        """
        
        Args:
            cert: np.ndarray
            compls: np.ndarray

        Returns:

        """
        cdef:
            Py_ssize_t i, j, max_i, max_j
            bint maximize_density
            double dens_cert
            np.ndarray degrees, set1, set2
            DTYPE_ADJ_t[::1] sum_adj
            DTYPE_ADJ_t[:, ::1] adj


        max_i = cert.size
        max_j = cert.size
        sum_adj = np.zeros(max_i, dtype=DTYPE_ADJ)
        adj = self.adjacency
        for i in range(max_i):
            for j in range(max_j):
                sum_adj[i] += adj[i][j]

        dens_cert = self._density(self.adjacency, cert, cert, True)

        # degrees = np.sum(self.adjacency[np.ix_(cert, cert)], axis=1).argsort()[::-1]
        degrees = np.argsort(sum_adj)[::-1]

        # print(degrees)
        # print(np.sum(self.adjacency[np.ix_(cert, cert)], axis=1).argsort)
        # if self.verbose:
        #     print(f'dens cert: {dens_cert}')
        #     print('degs', degrees)
        if dens_cert > self.threshold:
            set1 = cert[degrees[0::2]]
            set2 = cert[degrees[1::2]]
            maximize_density = True
        else:
            # if self.verbose:
            #     print('random sets')
            #     print(f'cert {cert}, cert size {cert.size}')
            #     print(f'--- numpy random state {np.random.get_state()}')
            set1 = np.random.choice(cert, cert.size // 2, replace=False)
            # if self.verbose:
            #     print(f'set1 : {set1}')
            set2 = np.setdiff1d(cert, set1)
            # if self.verbose:
            #     print(f'set2 : {set2}')
            maximize_density = False
            #
            # if self.verbose:
            #     print(f'certificat compls', compls)
            #     print('before sets')
            #     print(set1)
            #     print(set2)

        set1, compls = self._fill_new_set(set1, compls, maximize_density)
        set2, compls = self._fill_new_set(set2, compls, maximize_density)

        # if self.verbose:
        #     print('after sets')
        #     print(set1)
        #     print(set2)
        #     print('|||||||')
        return set1, set2, compls

    @cython.profile(True)
    cpdef tuple _fill_new_set(self, np.ndarray new_set, np.ndarray compls, bint maximize_density):
        """ Find nodes that can be added
        Move from compls the nodes in can_be_added until we either finish the nodes or reach the desired cardinality
        :param new_set: np.array(), array of indices of the set that must be augmented
        :param compls: np.array(), array of indices used to augment the new_set
        :param maximize_density: bool, used to augment or decrement density
        """
        cdef:
            Py_ssize_t i, j, max_i, max_j
            Py_ssize_t idx_i, idx_j
            DTYPE_ADJ_t[:, ::1] nodes_, adj
            DTYPE_UINT_t[:, ::1] tile_
            DTYPE_UINT_t[:, ::1] to_add_
            DTYPE_UINT_t[::1] counter
            double val

        val = 1.0 if maximize_density else 0.0
        nodes = self.adjacency[np.ix_(new_set, compls)] == val

        adj = self.adjacency

        compls.sort()

        # max_i = new_set.size
        # max_j = compls.size
        # nodes_ = np.zeros((max_i, max_j), dtype=DTYPE_ADJ)
        # tile_ = np.zeros((max_i, max_j), dtype=DTYPE_UINT)
        # counter = np.zeros(max_j, dtype=DTYPE_UINT)
        # for i in range(max_i):
        #     idx_i = new_set[i]
        #
        #     for j in range(max_j):
        #         idx_j = compls[j]
        #         nodes_[i][j] = adj[idx_i][idx_j] == val
        #         tile_[i][j] = compls[j]
        #
        #         if adj[idx_i][idx_j] == val:
        #             counter[j] += 1
        #
        # print(f'{counter.base}')


        # print(f'max_i: {new_set.size}')
        # print(f'max_j: {compls.size}')

        # print(nodes.shape)
        # print(nodes[0])
        # print(nodes_.base[0])
        # assert np.array_equal(nodes, np.asarray(nodes_))
        # assert np.array_equal(np.tile(compls, (len(new_set), 1)), tile_)

        # print(np.tile(compls, (len(new_set), 1))[nodes].shape)
        # print(np.tile(compls, (len(new_set), 1)).shape)
        # print(np.sum(nodes, axis=0))
        # to_add = np.unique(np.tile(compls, (len(new_set), 1))[nodes], return_counts=True)
        # print(f'{to_add[1]}')
        # assert np.array_equal(compls, to_add[0]), f'compls not equal'
        # assert np.array_equal(to_add[1], counter), f'counter not equal'
        # to_add = to_add[0][to_add[1].argsort()]
        to_add = compls[np.sum(nodes, axis=0).argsort()]

        # print(f'to add shape {to_add.shape}')
        if not maximize_density:
            to_add = to_add[::-1]

        cdef list tmp_new_set = list(new_set)
        # print(compls)
        # print(np.sum(nodes, axis=0).argsort())
        # print(to_add)

        # while new_set.size < self.partition_size:
        while len(tmp_new_set) < self.partition_size:

            # If there are nodes in to_add, we keep moving from compls to new_set
            if to_add.size > 0:
                node, to_add = to_add[-1], to_add[:-1]
                # new_set = np.append(new_set, node)
                tmp_new_set.append(node)
                # print(f'node {node}')
                # print(f'idx', np.argwhere(compls == node))
                # compls = np.delete(compls, np.argwhere(compls == node))
                compls = compls[:-1]
            else:
                # If there aren't candidate nodes, we keep moving from complements
                # to certs until we reach the desired cardinality
                node, compls = compls[-1], compls[:-1]
                # new_set = np.append(new_set, node)
                tmp_new_set.append(node)

        return np.array(tmp_new_set), compls

    @cython.profile(True)
    cpdef void _update_partitions(self, np.ndarray part1, np.ndarray part2):
        self._update_partition(part1)
        self._update_partition(part2)

    @cython.profile(True)
    cpdef void _update_partition(self, np.ndarray partition):
        self.partition_idx -= 1
        self.new_dict_partitions[self.partition_idx] = partition

        if partition.size > self.partition_size:
            last, self.new_dict_partitions[self.partition_idx] = self.new_dict_partitions[self.partition_idx][-1], \
                                                                 self.new_dict_partitions[self.partition_idx][:-1]
            self.new_dict_partitions[0].append(last)
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            # print(last, self.new_dict_partitions)
