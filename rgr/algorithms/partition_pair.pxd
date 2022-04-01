cimport numpy as np

from rgr.constants.types cimport DTYPE_ADJ_t, DTYPE_UINT

cdef class PartitionPair:
    cdef:
        readonly:
            int prts_size
            int r, s
            int bip_sum_edges
            double bip_avg_deg, bip_density
            double eps
            np.ndarray adjacency, bip_adj
            np.ndarray s_indices, r_indices
            np.ndarray s_degrees, r_degrees