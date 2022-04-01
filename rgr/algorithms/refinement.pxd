cimport numpy as np

from rgr.algorithms.certificates_complements cimport CertificatesComplements
from rgr.constants.types cimport DTYPE_ADJ_t, DTYPE_IDX


cdef class Refinement:
    cdef:
        int n_nodes, n_partitions, partition_size
        int partition_idx
        double epsilon, threshold
        list partitions, certificates_complements
        readonly dict new_partitions
        np.ndarray pair_densities
        np.ndarray adjacency


    cpdef double _density(self,
                          np.ndarray adjacency,
                          np.ndarray indices_a,
                          np.ndarray indices_b)

    cpdef np.ndarray _pairwise_densities(self)

    cpdef bint _refinement_degree_based(self)

    cpdef int _choose_candidate(self, int s, list irregulars)

    cdef double _density_candidate(self,
                                   np.ndarray s_indices,
                                   np.ndarray r_indices,
                                   double s_dens,
                                   int r)
    cpdef tuple _update_certs(self,
                              np.ndarray cert,
                              np.ndarray compls)

    cpdef tuple _fill_new_set(self, new_set, compls, maximize_density)

    cpdef void _update_partitions(self, np.ndarray part1, np.ndarray part2)

    cpdef void _update_partition(self, np.ndarray partition)
