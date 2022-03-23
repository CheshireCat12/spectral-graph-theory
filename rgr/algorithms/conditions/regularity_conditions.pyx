import numpy as np
cimport numpy as np

cdef class RegularityConditions:

    def __init__(self, int[:, ::1] adjacency, int[::1] classes, int r, int s, float eps):
        self.adjacency = adjacency
        self.classes = classes
        self.r = r
        self.s = s
        self.eps = eps

        self.s_indices = np.where(np.asarray(classes) == s)[0]
        self.r_indices = np.where(np.asarray(classes) == r)[0]

        # Bipartite adjacency matrix
        bip_adj = np.asarray(adjacency)[np.ix_(self.s_indices, self.r_indices)]

        self.cls_cardinality = bip_adj.shape[0]

        # The sum of edges is equal to the sum of degrees in bipartite graphs
        bip_sum_edges = np.sum(bip_adj)

        # Bipartite average degree
        self.bip_avg_deg = bip_sum_edges / self.cls_cardinality
        # Bipartite edge density
        self.bip_edge_density = bip_sum_edges / (self.cls_cardinality ** 2)

        self.r_degrees = np.sum(bip_adj, axis=0)
        self.s_degrees = np.sum(bip_adj, axis=1)


    @property
    def threshold_dev(self):
        return self.eps**4 * self.cls_cardinality

    cpdef tuple conditions(self):
        # s, r certificates
        certificates = [[], []]
        # s, r complements
        complements = [[], []]

        if self.condition_1(certificates, complements):
            return True, certificates, complements
        elif self.condition_2(certificates, complements):
            pass

    cpdef bint condition_1(self, list certificates, list complements):
        is_regular = self.bip_avg_deg < (self.eps**3) * self.cls_cardinality

        return is_regular

    cpdef bint condition_2(self, list certificates, list complements):
        threshold_n_nodes = (1/8) * self.threshold_dev

        is_irregular = False

        deviated_nodes_mask = np.abs(np.asarray(self.s_degrees)-self.bip_avg_deg) >= self.threshold_dev

        if np.sum(deviated_nodes_mask) > threshold_n_nodes:
            is_irregular = True

            s_certs = self.s_indices[deviated_nodes_mask]
            s_complements = np.setdiff1d(self.s_indices, s_certs)

            b_mask = np.asarray(self.adjacency)[np.ix_(s_certs, self.r_indices)] > 0
            b_mask = b_mask.any(0)

            r_certs = self.r_indices[b_mask]
            r_complements = np.setdiff1d(self.r_indices, r_certs)

            certificates = [s_certs.tolist(), r_certs.tolist()]
            complements = [s_complements.tolist(), r_complements.tolist()]

        return is_irregular

    cpdef bint condition_3(self, list certificates, list complements):
        return False