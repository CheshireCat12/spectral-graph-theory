import numpy as np

cdef class PartitionPair:

    def __init__(self, np.ndarray adjacency, np.ndarray s_indices, np.ndarray r_indices, double eps):
        """

        Args:
            adjacency:
            s_indices:
            r_indices:
            eps:
        """

        self.adjacency = adjacency
        self.s_indices = s_indices
        self.r_indices = r_indices
        self.eps = eps

        # Bipartite adjacency matrix
        self.bip_adj = adjacency[np.ix_(s_indices, r_indices)]

        # Cardinality of the partitions
        self.prts_size = self.bip_adj.shape[0]

        self.bip_sum_edges = np.sum(self.bip_adj)

        # Bipartite average degree
        # To have a faster summation of the bipartite degrees
        # I directly sum the elements over the whole matrix,
        # so I don't have to divide the sum by 2
        self.bip_avg_deg = self.bip_sum_edges / self.prts_size

        self.s_degrees = np.sum(self.bip_adj, axis=1)
        self.r_degrees = np.sum(self.bip_adj, axis=0)


# np.sum(np.sum(bip_adj, axis=0) + np.sum(bip_adj, axis=1)) / (2 * cls_cardinality)
# # Bipartite edge density
# bip_edge_density = bip_sum_edges / (cls_cardinality**2)
