import numpy as np
cimport numpy as np

cdef class RegularityConditions:
    def __init__(self, PartitionPair pair):
        self.pair = pair

    @property
    def threshold_dev(self):
        return self.pair.eps ** 4 * self.pair.prts_size

    cpdef tuple conditions(self):
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
        else:
            return False, None, None

    cpdef bint condition_1(self, list certificates, list complements):
        """
        
        Args:
            certificates: 
            complements: 

        Returns:

        """
        is_regular = self.pair.bip_avg_deg < (self.pair.eps ** 3) * self.pair.prts_size

        return is_regular

    cpdef bint condition_2(self, list certificates, list complements):
        threshold_n_nodes = (1 / 8) * self.threshold_dev

        is_irregular = False

        deviated_nodes_mask = np.abs(self.pair.s_degrees - self.pair.bip_avg_deg) >= self.threshold_dev

        if np.sum(deviated_nodes_mask) > threshold_n_nodes:
            is_irregular = True

            s_certs = self.pair.s_indices[deviated_nodes_mask]
            s_complements = np.setdiff1d(self.pair.s_indices, s_certs)

            b_mask = self.pair.adjacency[np.ix_(s_certs, self.pair.r_indices)] > 0
            b_mask = b_mask.any(0)

            r_certs = self.pair.r_indices[b_mask]
            r_complements = np.setdiff1d(self.pair.r_indices, r_certs)

            certificates[:] = [r_certs.tolist(), s_certs.tolist()]
            complements[:] = [r_complements.tolist(), s_complements.tolist()]

        return is_irregular

    cpdef bint condition_3(self, list certificates, list complements):
        is_irregular = False
        # print(f'bip_adj {self.pair.bip_adj}')

        ngh_dev = neighbourhood_deviation(self.pair.bip_adj,
                                          self.pair.bip_avg_deg,
                                          self.pair.prts_size)
        print(f'neigh def {ngh_dev}')

        yp_filter = find_Yp(self.pair.s_degrees,
                            self.pair.bip_avg_deg,
                            self.pair.prts_size,
                            self.pair.eps)
        print('yp_filter')
        print(yp_filter)

        if yp_filter.size == 0:
            is_irregular = True
            return is_irregular

        s_certs, y0 = compute_y0(ngh_dev,
                                 self.pair.s_indices,
                                 yp_filter,
                                 self.pair.prts_size,
                                 self.pair.eps)

        print(s_certs)
        if s_certs is None:
            is_irregular = False
            return is_irregular
        else:
            assert np.array_equal(np.intersect1d(s_certs, self.pair.s_indices),
                                  s_certs) == True, "cert_is not subset of s_indices"
            assert (y0 in self.pair.s_indices) == True, "y0 not in s_indices"

            is_irregular = True
            b_mask = self.pair.adjacency[np.ix_(np.array([y0]), self.pair.r_indices)] > 0
            r_certs = self.pair.r_indices[b_mask[0]]
            assert np.array_equal(np.intersect1d(r_certs, self.pair.r_indices),
                                  r_certs) == True, "cert_is not subset of s_indices"

            # [BUG] cannot do set(s_indices) - set(s_certs)
            s_complements = np.setdiff1d(self.pair.s_indices, s_certs)
            r_complements = np.setdiff1d(self.pair.r_indices, r_certs)
            assert s_complements.size + s_certs.size == self.pair.prts_size, "Wrong cardinality"
            assert r_complements.size + r_certs.size == self.pair.prts_size, "Wrong cardinality"

            certificates[:] = [r_certs.tolist(), s_certs.tolist()]
            complements[:] = [r_complements.tolist(), s_complements.tolist()]
            return is_irregular

cpdef find_Yp(s_degrees, bip_avg_deg, cls_cardinality, eps):
    threshold_deviation = (eps ** 4) * cls_cardinality
    mask = np.abs(s_degrees - bip_avg_deg) < threshold_deviation
    yp_i = np.where(mask == True)[0]

    return yp_i

cpdef compute_y0(ngh_dev, s_indices, yp_i, cls_cardinality, eps):
    threshold_dev = 2 * eps**4 * cls_cardinality
    rect_mat = ngh_dev[yp_i]

    boolean_mat = rect_mat > threshold_dev
    cardinality_by0s = np.sum(boolean_mat, axis=1)

    y0_idx = np.argmax(cardinality_by0s)
    aux = yp_i[y0_idx]

    y0 = s_indices[aux]

    if cardinality_by0s[y0_idx] > (eps ** 4 * cls_cardinality / 4.0):
        cert_s = s_indices[boolean_mat[y0_idx]]
        return cert_s, y0
    else:
        return None, y0


cpdef neighbourhood_deviation(bip_adj, bip_avg_deg, cls_cardinality):
    mat = np.matmul(np.asarray(bip_adj.T), np.asarray(bip_adj))
    print(mat)
    mat = mat - (bip_avg_deg**2) / cls_cardinality

    return mat
