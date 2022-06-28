cimport numpy as np

from rgr.algorithms.certificates_complements cimport CertificatesComplements
from rgr.constants.types cimport DTYPE_ADJ_t, DTYPE_ADJ, DTYPE_IDX_t, DTYPE_IDX, DTYPE_UINT_t, DTYPE_UINT


cdef class Refinement:
    cdef:
        readonly:
            bint is_refined
            int n_partitions
            list new_partitions
            dict new_dict_partitions
            np.ndarray pair_densities
        bint verbose
        int n_nodes, partition_idx, partition_size
        double epsilon, threshold
        list partitions, certificates_complements
        np.ndarray adjacency


    cpdef double _density(self,
                          DTYPE_ADJ_t[:, ::1] adjacency,
                          DTYPE_UINT_t[::1] indices_a,
                          DTYPE_UINT_t[::1] indices_b,
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

    cpdef tuple _fill_new_set(self, np.ndarray new_set, np.ndarray compls, bint maximize_density)

    cpdef void _update_partitions(self, np.ndarray part1, np.ndarray part2)

    cpdef void _update_partition(self, np.ndarray partition)
