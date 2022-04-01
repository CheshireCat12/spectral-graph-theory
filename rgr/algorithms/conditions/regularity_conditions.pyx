import numpy as np
cimport cython

cdef class RegularityConditions:
    """Check if the Alon's regularity conditions are fulfilled."""

    def __init__(self, PartitionPair pair):
        """

        Args:
            pair: PartitionPair
        """
        self.pair = pair

        # Threshold of the deviation of the degree from the average bipartite degree
        self.threshold_dev = self.pair.eps ** 4 * self.pair.prts_size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple conditions(self):
        cdef:
            bint is_condition_verified
            list conditions_to_verify
            CertificatesComplements certs_compls

        conditions_to_verify = [
            self.condition_1,
            self.condition_2,
            self.condition_3,
        ]

        for idx, condition in enumerate(conditions_to_verify):
            is_condition_verified, certs_compls = condition()

            if is_condition_verified:
                return is_condition_verified, certs_compls
        else:
            return False, CertificatesComplements()

    cpdef tuple condition_1(self):
        """
        
        Returns:

        """
        cdef:
            bint is_regular

        is_regular = self.pair.bip_avg_deg < (self.pair.eps ** 3) * self.pair.prts_size

        return is_regular, CertificatesComplements()

    cpdef tuple condition_2(self):
        """
        
        Returns:

        """
        cdef:
            bint is_irregular
            double threshold_n_nodes
            list certificates, complements
            np.ndarray deviated_nodes_mask, b_mask
            np.ndarray s_certs, r_certs
            np.ndarray s_complements, r_complements

        threshold_n_nodes = (1 / 8) * self.threshold_dev

        is_irregular = False
        certificates, complements = None, None

        deviated_nodes_mask = np.abs(self.pair.s_degrees - self.pair.bip_avg_deg) >= self.threshold_dev

        if np.sum(deviated_nodes_mask) > threshold_n_nodes:
            is_irregular = True

            s_certs = self.pair.s_indices[deviated_nodes_mask]
            s_complements = np.setdiff1d(self.pair.s_indices, s_certs)

            b_mask = self.pair.adjacency[np.ix_(s_certs, self.pair.r_indices)] > 0
            b_mask = b_mask.any(0)

            r_certs = self.pair.r_indices[b_mask]
            r_complements = np.setdiff1d(self.pair.r_indices, r_certs)

            certificates = [r_certs, s_certs]
            complements = [r_complements, s_complements]

        return is_irregular, CertificatesComplements(certificates, complements)

    cpdef tuple condition_3(self):
        """
        
        Returns:

        """
        cdef:
            bint is_irregular
            list certificates, complements
            np.ndarray deviated_nodes_mask, b_mask
            np.ndarray s_certs, r_certs
            np.ndarray s_complements, r_complements
            np.ndarray ngh_dev
            np.ndarray yp_filter
            DTYPE_IDX_t y0

        is_irregular = False
        certificates, complements = None, None

        ngh_dev = neighbourhood_deviation(self.pair.bip_adj,
                                          self.pair.bip_avg_deg,
                                          self.pair.prts_size)

        yp_filter = find_Yp(self.pair.s_degrees,
                            self.pair.bip_avg_deg,
                            self.threshold_dev)

        if yp_filter.size == 0:
            is_irregular = True

            return is_irregular, CertificatesComplements()

        s_certs, y0 = compute_y0(ngh_dev,
                                 self.pair.s_indices,
                                 yp_filter,
                                 self.pair.prts_size,
                                 self.pair.eps)

        if s_certs is not None:
            assert np.array_equal(np.intersect1d(s_certs, self.pair.s_indices),
                                  s_certs) == True, "cert_is not subset of s_indices"
            assert y0 in self.pair.s_indices, "y0 not in s_indices"

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

            certificates = [r_certs, s_certs]
            complements = [r_complements, s_complements]

        return is_irregular, CertificatesComplements(certificates, complements)

cpdef np.ndarray neighbourhood_deviation(np.ndarray[DTYPE_ADJ_t, ndim=2] bip_adj,
                                         double bip_avg_deg,
                                         int cls_cardinality):
    """

    Args:
        bip_adj: np.ndarray[DTYPE_ADJ_t, ndim=2]
        bip_avg_deg: double
        cls_cardinality: int

    Returns:
        np.ndarray[DTYPE_FLOAT_t, ndim=2]
    """
    cdef:
        np.ndarray[DTYPE_FLOAT_t, ndim=2] mat

    mat = np.matmul(bip_adj.T, bip_adj).astype(DTYPE_FLOAT)
    mat = mat - (bip_avg_deg ** 2) / cls_cardinality

    return mat

cpdef np.ndarray find_Yp(np.ndarray[DTYPE_UINT_t, ndim=1] s_degrees,
                         double bip_avg_deg,
                         double threshold_dev):
    """
    
    Args:
        s_degrees: 
        bip_avg_deg: double
        threshold_dev: double

    Returns:

    """

    mask = np.abs(s_degrees - bip_avg_deg) < threshold_dev
    yp_i = np.where(mask == True)[0]

    return yp_i

cpdef tuple compute_y0(np.ndarray[DTYPE_FLOAT_t, ndim=2] ngh_dev,
                       np.ndarray[DTYPE_IDX_t, ndim=1] s_indices,
                       yp_i,
                       cls_cardinality,
                       double eps):
    threshold_dev = 2 * eps ** 4 * cls_cardinality
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
