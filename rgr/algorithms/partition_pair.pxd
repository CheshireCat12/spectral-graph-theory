cimport cython
cimport numpy as np

from rgr.constants.types cimport DTYPE_ADJ, DTYPE_ADJ_t, DTYPE_UINT, DTYPE_UINT_t

cdef class PartitionPair:
    cdef:
        readonly:
            int r, s
            int bip_sum_edges, prts_size
            double bip_avg_deg, bip_density
            double eps
            np.ndarray adjacency, bip_adj
            np.ndarray s_indices, r_indices
            np.ndarray s_degrees, r_degrees

cdef class PartitionPairFast:
    cdef:
        readonly:
            int r, s
            int bip_sum_edges, prts_size
            double bip_avg_deg, bip_density
            double eps
            np.ndarray adjacency, bip_adj
            DTYPE_ADJ_t[:, ::1] adjacency_, bip_adj_
            np.ndarray s_indices, r_indices
            DTYPE_UINT_t[::1] s_indices_, r_indices_
            np.ndarray s_degrees, r_degrees
