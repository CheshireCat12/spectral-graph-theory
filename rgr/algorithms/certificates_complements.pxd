cimport numpy as np

from rgr.constants.types cimport DTYPE_IDX

cdef class CertificatesComplements:

    cdef:
        list certificates
        list complements