import numpy as np

cdef class PartitionPair:
    """Data class used to handle information between a pair of partitions."""

    def __init__(self,
                 np.ndarray[DTYPE_ADJ_t, ndim=2] adjacency,
                 list partitions,
                 int r,
                 int s,
                 double eps):
        """

        Args:
            adjacency: np.ndarray[DTYPE_ADJ_t, ndim=2]
            partitions: List[np.ndarray[DTYPE_IDX_t, ndim=1]
            r: int
            s: int
            eps: double
        """

        self.adjacency = adjacency
        self.r = r
        self.s = s
        self.eps = eps

        self.r_indices = partitions[r]
        self.s_indices = partitions[s]

        # Bipartite adjacency matrix
        self.bip_adj = self.adjacency[np.ix_(self.s_indices, self.r_indices)]

        # Cardinality of the partitions
        self.prts_size = len(self.s_indices) #self.bip_adj.shape[0]

        self.bip_sum_edges = np.sum(self.bip_adj)

        # Bipartite average degree
        # To have a faster summation of the bipartite degrees
        # I directly sum the elements over the whole matrix,
        # so I don't have to divide the sum by 2
        self.bip_avg_deg = self.bip_sum_edges / self.prts_size

        self.bip_density = self.bip_sum_edges / (self.prts_size**2)

        self.s_degrees = np.sum(self.bip_adj, axis=1)
        self.r_degrees = np.sum(self.bip_adj, axis=0)
