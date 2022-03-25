cimport numpy as np

cdef class PartitionPair:

    cdef:
        int prts_size
        double eps
        np.ndarray adjacency
        np.ndarray bip_avg_deg
        np.ndarray s_indices, r_indices
        np.ndarray s_degrees, r_degrees