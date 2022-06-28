import numpy as np
cimport numpy as np
cimport cython


cdef class CertificatesComplements:

    @cython.nonecheck(False)
    def __init__(self, list certificates=None, list complements=None):
        """

        Args:
            certificates: List[np.ndarray[DTYPE_IDX_t, ndim=1]]
                List containing the certificates for idx r and s (e.g., [r_certs, s_certs])
            complements: List[np.ndarray[DTYPE_IDX_t, ndim=1]]
                List containing the complements of the certificates
                for idx r and s (e.g., [r_compls, s_compls])
        """
        if certificates is None:
            certificates = [np.array([], dtype=DTYPE_UINT)] * 2
        if complements is None:
            complements = [np.array([], dtype=DTYPE_UINT)] * 2

        self.certificates = certificates
        self.complements = complements

        self.r_certs = certificates[0]
        self.s_certs = certificates[1]

        self.r_compls = complements[0]
        self.s_compls = complements[1]

    # @property
    # def r_certs(self):
    #     return self.certificates[0]
    #
    # @property
    # def s_certs(self):
    #     return self.certificates[1]
    #
    # @property
    # def r_compls(self):
    #     return self.complements[0]
    #
    # @property
    # def s_compls(self):
    #     return self.complements[1]

    def __repr__(self):
        return f'Certificates {self.certificates}' \
               f'Complements {self.complements}'

    cpdef bint is_r_certificate_defined(self):
        return self.r_certs.size > 0