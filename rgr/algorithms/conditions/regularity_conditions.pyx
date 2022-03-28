import numpy as np
cimport numpy as np

cdef class RegularityConditions:

    def __init__(self):
       self.pair = None

    @property
    def _threshold_dev(self):
        return self.pair.eps**4 * self.pair.cls_cardinality

    @staticmethod
    def conditions(self, pair):
        self.pair = pair

        # s, r certificates
        certificates = [[], []]
        # s, r complements
        complements = [[], []]

        if self.condition_1(certificates, complements):
            return True, certificates, complements
        elif self.condition_2(certificates, complements):
            return True, certificates, complements
        elif self.condition_3(certificates, complements):
            return True, certificates, complements

    cpdef bint condition_1(self, list certificates, list complements):
        """
        
        Args:
            certificates: 
            complements: 

        Returns:

        """
        is_regular = self.pair.bip_avg_deg < (self.pair.eps**3) * self.pair.cls_cardinality

        return is_regular

    cpdef bint condition_2(self, list certificates, list complements):
        threshold_n_nodes = (1/8) * self._threshold_dev

        is_irregular = False

        deviated_nodes_mask = np.abs(self.pair.s_degrees-self.pair.bip_avg_deg) >= self._threshold_dev

        if np.sum(deviated_nodes_mask) > threshold_n_nodes:
            is_irregular = True

            s_certs = self.pair.s_indices[deviated_nodes_mask]
            s_complements = np.setdiff1d(self.pair.s_indices, s_certs)

            b_mask = self.pair.adjacency[np.ix_(s_certs, self.pair.r_indices)] > 0
            b_mask = b_mask.any(0)

            r_certs = self.pair.r_indices[b_mask]
            r_complements = np.setdiff1d(self.pair.r_indices, r_certs)

            certificates = [s_certs.tolist(), r_certs.tolist()]
            complements = [s_complements.tolist(), r_complements.tolist()]

        return is_irregular

    cpdef bint condition_3(self, list certificates, list complements):
        return False