cimport numpy as np

cdef class PartitionPair:
    cdef:
        public:
            int prts_size
            int r, s
            int bip_sum_edges
            double bip_avg_deg, bip_density
            double eps
            np.ndarray adjacency, bip_adj
            np.ndarray s_indices, r_indices
            np.ndarray s_degrees, r_degrees
