cimport numpy as np

from rgr.algorithms.certificates_complements cimport CertificatesComplements
from rgr.constants.types cimport DTYPE_ADJ_t, DTYPE_IDX


cdef class Refinement:
    cdef:
        bint verbose
        readonly bint is_refined
        int n_nodes, partition_idx, partition_size
        readonly int n_partitions
        double epsilon, threshold
        list partitions, certificates_complements
        readonly list new_partitions
        readonly dict new_dict_partitions
        np.ndarray adjacency
        readonly np.ndarray pair_densities


    cpdef double _density(self,
                          np.ndarray adjacency,
                          np.ndarray indices_a,
                          np.ndarray indices_b,
                          bint same_index_set=*)

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
