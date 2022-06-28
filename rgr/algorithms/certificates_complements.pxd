cimport numpy as np

from rgr.constants.types cimport DTYPE_UINT_t, DTYPE_UINT

cdef class CertificatesComplements:

    cdef:
        list certificates
        list complements

        readonly:
            DTYPE_UINT_t[::1] r_certs, s_certs
            DTYPE_UINT_t[::1] r_compls, s_compls

    cpdef bint is_r_certificate_defined(self)