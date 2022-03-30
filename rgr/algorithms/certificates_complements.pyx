import numpy as np


cdef class CertificatesComplements:

    def __init__(self, list certificates=None, list complements=None):
        """

        Args:
            certificates: list
                List containing the certificates for idx r and s (e.g., [r_certs, s_certs])
            complements: list
                List containing the complements of the certificates
                for idx r and s (e.g., [r_compls, s_compls])
        """
        if certificates is None:
            certificates = [np.array([], dtype=DTYPE_IDX)] * 2
        if complements is None:
            complements = [np.array([], dtype=DTYPE_IDX)] * 2

        self.certificates = certificates
        self.complements = complements

    @property
    def r_certs(self):
        return self.certificates[0]

    @property
    def s_certs(self):
        return self.certificates[1]

    @property
    def r_compls(self):
        return self.complements[0]

    @property
    def s_compls(self):
        return self.complements[1]

    def __repr__(self):
        return f'Certificates {self.certificates}\n' \
               f'Complements {self.complements}'