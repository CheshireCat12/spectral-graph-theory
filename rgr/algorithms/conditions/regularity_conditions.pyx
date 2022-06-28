import numpy as np
cimport numpy as np
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
        self.threshold_dev = (self.pair.eps ** 4) * self.pair.prts_size

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple conditions(self):
        """
        Iterates over the alon's regularity conditions.
        
        Returns:

        """
        cdef:
            bint is_condition_verified
            list conditions_to_verify
            CertificatesComplements certs_compls

        conditions_to_verify = [
            self.condition_1,
            self.condition_3,
            self.condition_2,
        ]

        for idx, condition in enumerate(conditions_to_verify):
            is_condition_verified, certs_compls = condition()

            if is_condition_verified:
                # print(f'cond num: {idx}')
                return is_condition_verified, certs_compls
        else:
            return False, CertificatesComplements()

    @cython.profile(True)
    cpdef tuple condition_1(self):
        """
        
        Returns:

        """
        cdef:
            bint is_regular

        is_regular = self.pair.bip_avg_deg < (self.pair.eps ** 3) * self.pair.prts_size

        return is_regular, CertificatesComplements()

    @cython.profile(True)
    cpdef tuple condition_2(self):
        """
        
        Returns:

        """
        cdef:
            bint is_cond_verified
            double threshold_n_nodes
            list certificates, complements
            np.ndarray deviated_nodes_mask, b_mask
            np.ndarray s_certs, r_certs
            np.ndarray s_complements, r_complements

        threshold_n_nodes = (1 / 8) * self.threshold_dev

        is_cond_verified = False
        certificates, complements = None, None

        deviated_nodes_mask = np.abs(np.asarray(self.pair.s_degrees) - self.pair.bip_avg_deg) >= self.threshold_dev

        # print(f'deviated node mask: {np.sum(deviated_nodes_mask)}')
        # print(f'size_cls: {self.pair.prts_size}')
        # print(f'threshold dev: {threshold_n_nodes}')
        if np.sum(deviated_nodes_mask) > threshold_n_nodes:
            is_cond_verified = True

            s_certs = np.asarray(self.pair.s_indices)[deviated_nodes_mask]
            s_complements = np.setdiff1d(self.pair.s_indices, s_certs)

            b_mask = np.asarray(self.pair.adjacency)[np.ix_(s_certs, self.pair.r_indices)] > 0
            b_mask = b_mask.any(0)

            r_certs = np.asarray(self.pair.r_indices)[b_mask]
            r_complements = np.setdiff1d(self.pair.r_indices, r_certs)

            certificates = [r_certs, s_certs]
            complements = [r_complements, s_complements]

        return is_cond_verified, CertificatesComplements(certificates, complements)

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple condition_3(self):
        """
        
        Returns:

        """
        cdef:
            Py_ssize_t i, max_i
            bint is_irregular
            int idx
            list certificates, complements
            list r_certs_tmp, r_compls_tmp
            DTYPE_UINT_t y0
            DTYPE_UINT_t[::1] r_indices
            DTYPE_ADJ_t[:, ::1] adj
            DTYPE_FLOAT_t[:, ::1] ngh_dev
            DTYPE_UINT_t[::1] yp_filter
            DTYPE_UINT_t[::1] s_certs, s_complements
            DTYPE_UINT_t[::1] r_certs, r_complements


        is_irregular = False
        certificates, complements = None, None

        # ngh_dev = neighbourhood_deviation(self.pair.bip_adj,
        #                                   self.pair.bip_avg_deg,
        #                                   self.pair.prts_size)
        ngh_dev = neighbourhood_deviation_fast(self.pair.bip_adj,
                                          self.pair.bip_avg_deg,
                                          self.pair.prts_size)

        # yp_filter = find_Yp(np.asarray(self.pair.s_degrees),
        #                     self.pair.bip_avg_deg,
        #                     self.threshold_dev)

        yp_filter = find_Yp_fast(self.pair.s_degrees,
                                 self.pair.bip_avg_deg,
                                 self.threshold_dev)
        print(yp_filter)
        if yp_filter.size == 0:
            is_irregular = True

            return is_irregular, CertificatesComplements()

        # s_certs, y0 = compute_y0(ngh_dev,
        #                          np.asarray(self.pair.s_indices),
        #                          yp_filter,
        #                          self.threshold_dev)

        s_certs, s_complements, y0 = compute_y0_fast(ngh_dev,
                                                     self.pair.s_indices,
                                                     yp_filter,
                                                     self.threshold_dev)
        if s_certs is not None:
            # assert np.array_equal(np.intersect1d(s_certs, self.pair.s_indices),
            #                       s_certs) == True, "cert_is not subset of s_indices"
            # assert y0 in self.pair.s_indices, "y0 not in s_indices"
            print('s_cert exist')
            is_irregular = True
            r_indices = self.pair.r_indices
            adj = self.pair.adjacency
            max_i = r_indices.shape[0]
            # b_mask = np.zeros(max_i, dtype=np.int8)
            r_certs_tmp = []
            r_compls_tmp = []
            for i in range(max_i):
                idx = r_indices[i]
                if adj[y0][idx] > 0:
                    r_certs_tmp.append(idx)
                else:
                    r_compls_tmp.append(idx)
            print('toto')
            # print(f'new {b_mask}')
            # b_mask = np.asarray(self.pair.adjacency)[np.ix_(np.array([y0]), self.pair.r_indices)] > 0
            # print(b_mask)
            # r_certs = np.asarray(self.pair.r_indices)[b_mask[0]]
            r_certs = np.asarray(r_certs_tmp, dtype=DTYPE_UINT)
            r_complements = np.asarray(r_compls_tmp, dtype=DTYPE_UINT)
            # assert np.array_equal(np.intersect1d(r_certs, np.asarray(self.pair.r_indices)),
            #                       r_certs) == True, "cert_is not subset of s_indices"

            # [BUG] cannot do set(s_indices) - set(s_certs)
            # s_complements = np.setdiff1d(np.asarray(self.pair.s_indices), s_certs)
            # assert np.array_equal(s_complements, s_complements_f), f'complements'
            # assert np.array_equal(s_certs, s_certs_f), 'f certificates'
            # if not np.array_equal(s_complements, s_compls_f):
            #     print(f'r {self.pair.r}, s {self.pair.s}')
            #
            #     print(f's_indices, {self.pair.s_indices}')
            #     print(f's_cert_f, {s_certs_f}')
            #     # print(f's_compls_f, {s_compls_f}')
            #     print(f's_cert, {s_certs}')
            #     print(f's_complements, {s_complements}')
            # r_complements = np.setdiff1d(np.asarray(self.pair.r_indices), r_certs)
            # assert s_complements.size + s_certs.size == self.pair.prts_size, "Wrong cardinality"
            # assert r_complements.size + r_certs.size == self.pair.prts_size, "Wrong cardinality"
            print('testsetsets')
            certificates = [r_certs, s_certs]
            complements = [r_complements, s_complements]

        return is_irregular, CertificatesComplements(certificates, complements)

@cython.profile(True)
cpdef np.ndarray neighbourhood_deviation(DTYPE_ADJ_t[:, ::1] bip_adj,
                                         double bip_avg_deg,
                                         int prts_cardinality):
    """

    Args:
        bip_adj: np.ndarray[DTYPE_ADJ_t, ndim=2]
        bip_avg_deg: double
        prts_cardinality: int

    Returns:
        np.ndarray[DTYPE_FLOAT_t, ndim=2]
    """
    cdef:
        np.ndarray[DTYPE_FLOAT_t, ndim=2] mat
        np.ndarray[DTYPE_FLOAT_t, ndim=2, mode='c'] bip_adj_f32

    bip_adj_f32 = np.array(bip_adj, dtype=DTYPE_FLOAT)
    # mat = np.matmul(bip_adj_.T, bip_adj_).astype(DTYPE_FLOAT)
    mat = np.matmul(bip_adj_f32.T, bip_adj_f32)
    mat -= (bip_avg_deg ** 2) / prts_cardinality

    return mat

from cython.parallel import prange
@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_FLOAT_t[:, ::1] neighbourhood_deviation_fast(DTYPE_ADJ_t[:, ::1] bip_adj,
                                                         double bip_avg_deg,
                                                         int prts_cardinality):
    """

    Args:
        bip_adj: np.ndarray[DTYPE_ADJ_t, ndim=2]
        bip_avg_deg: double
        prts_cardinality: int

    Returns:
        np.ndarray[DTYPE_FLOAT_t, ndim=2]
    """
    cdef:
        Py_ssize_t i, j
        Py_ssize_t max_i, max_j
        double offset
        DTYPE_FLOAT_t[:, ::1] mat
        np.ndarray[DTYPE_FLOAT_t, ndim = 2, mode = 'c'] bip_adj_f32


    bip_adj_f32 = np.array(bip_adj, dtype=DTYPE_FLOAT)
    mat = np.dot(bip_adj_f32.T, bip_adj_f32)

    # offset = pow(bip_avg_deg, 2) / prts_cardinality
    offset = (bip_avg_deg**2) / prts_cardinality

    max_i = mat.shape[0]
    max_j = mat.shape[1]

    for i in range(max_i):
    # for i in prange(max_i, nogil=True):
        for j in range(max_j):
            # mat[i][j] = mat[i][j] - offset
            mat[i][j] -= offset

    # print(np.array(mat))
    return mat

@cython.profile(True)
cpdef np.ndarray find_Yp(np.ndarray[DTYPE_UINT_t, ndim=1, mode='c'] s_degrees,
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
    # print(yp_i)
    return yp_i.astype(DTYPE_UINT)

from libc.math cimport abs, pow

@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_UINT_t[::1] c_find_Yp_fast(DTYPE_UINT_t[::1] s_degrees,
                                    double bip_avg_deg,
                                    double threshold_dev):
    """

    Args:
        s_degrees: 
        bip_avg_deg: double
        threshold_dev: double

    Returns:

    """
    cdef:
        Py_ssize_t i
        Py_ssize_t max_i = s_degrees.shape[0]
        list l_yp_i = []

    for i in range(max_i):
        if abs(s_degrees[i] - bip_avg_deg) < threshold_dev:
            l_yp_i.append(i)

    return np.array(l_yp_i, dtype=DTYPE_UINT)

def find_Yp_fast(s_degrees, bip_avg_deg, threshold_dev):
    return c_find_Yp_fast(s_degrees, bip_avg_deg, threshold_dev)

@cython.profile(True)
cpdef tuple compute_y0(np.ndarray[DTYPE_FLOAT_t, ndim=2, mode='c'] ngh_dev,
                       np.ndarray[DTYPE_IDX_t, ndim=1, mode='c'] s_indices,
                       np.ndarray[DTYPE_UINT_t, ndim=1, mode='c'] yp_i,
                       double threshold_dev):
    """
    
    Args:
        ngh_dev: 
        s_indices: 
        yp_i: 
        threshold_dev: 

    Returns:

    """
    # threshold_dev = 2 * eps ** 4 * cls_cardinality
    rect_mat = ngh_dev[yp_i]

    boolean_mat = rect_mat > 2 * threshold_dev
    cardinality_by0s = np.sum(boolean_mat, axis=1)

    y0_idx = np.argmax(cardinality_by0s)
    # print(f'naive max_sum {cardinality_by0s[y0_idx]}')
    # print(f'naive y0 {y0_idx}')
    aux = yp_i[y0_idx]

    y0 = s_indices[aux]

    if cardinality_by0s[y0_idx] > (threshold_dev / 4.0):
        cert_s = s_indices[boolean_mat[y0_idx]]
        # np.setdiff1d(s_indices, cert_s)
        return cert_s, y0
    else:
        return None, y0

@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple compute_y0_fast(DTYPE_FLOAT_t[:, ::1] ngh_dev,
                            DTYPE_UINT_t[::1] s_indices,
                            DTYPE_UINT_t[::1] yp_i,
                            double threshold_dev):
    """

    Args:
        ngh_dev:
        s_indices:
        yp_i:
        threshold_dev:

    Returns:

    """
    cdef:
        Py_ssize_t i, j
        Py_ssize_t max_i, max_j
        DTYPE_UINT_t[::1] cardinality_by0s
        DTYPE_ADJ_t[::1] boolean_vec, boolean_vec_tmp
        DTYPE_FLOAT_t[:, ::1] rect_mat

    rect_mat = np.asarray(ngh_dev)[yp_i]

    max_i = rect_mat.shape[0]
    max_j = rect_mat.shape[1]

    cdef int y0_idx = 0
    cdef int sum_, max_sum
    boolean_vec_tmp = np.zeros(max_j, dtype=DTYPE_ADJ)
    boolean_vec = np.zeros(max_j, dtype=DTYPE_ADJ)

    max_sum = -1

    for i in range(max_i):
        sum_ = 0
        # boolean_vec_tmp[:] = zero_vec
        boolean_vec_tmp[:] = 0
        # boolean_vec_tmp = np.zeros(max_j, dtype=DTYPE_ADJ)
        # print(boolean_vec_tmp.base)
        for j in range(max_j):
            boolean_vec_tmp[j] = rect_mat[i][j] > 2 * threshold_dev
            if rect_mat[i][j] > 2 * threshold_dev:
                sum_ += 1 #rect_mat[i][j]
                # cardinality_by0s[i] += rect_mat[i][j]

        if sum_ > max_sum:
            max_sum = sum_
            y0_idx = i
            boolean_vec[:] = boolean_vec_tmp

    # print(f'fast max sum {max_sum}')
    # print(f'fast y0 {y0_idx}')

    cdef DTYPE_UINT_t aux = yp_i[y0_idx]
    cdef DTYPE_UINT_t y0 = s_indices[aux]
    cdef list cert_s, compl_s
    cdef DTYPE_UINT_t idx
    if max_sum > (threshold_dev / 4):
        cert_s = []
        compl_s = []
        max_i = boolean_vec.shape[0]
        for i in range(max_i):
            idx = s_indices[i]
            if boolean_vec[i]:
                cert_s.append(idx)
            else:
                compl_s.append(idx)

        return np.array(cert_s, dtype=DTYPE_UINT), np.array(compl_s, dtype=DTYPE_UINT), y0
    return None, None, y0
    #
    #
    # boolean_mat = rect_mat > 2 * threshold_dev
    # cardinality_by0s = np.sum(boolean_mat, axis=1)
    #
    # y0_idx = np.argmax(cardinality_by0s)
    # aux = yp_i[y0_idx]
    #
    # y0 = s_indices[aux]
    #
    # if cardinality_by0s[y0_idx] > (threshold_dev / 4.0):
    #     cert_s = s_indices[boolean_mat[y0_idx]]
    #     return cert_s, y0
    # else:
