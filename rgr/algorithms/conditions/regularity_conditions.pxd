cimport cython
cimport numpy as np

from rgr.algorithms.certificates_complements cimport CertificatesComplements
from rgr.algorithms.partition_pair cimport PartitionPair, PartitionPairSlow
from rgr.constants.types cimport DTYPE_ADJ_t, DTYPE_ADJ, DTYPE_FLOAT_t, DTYPE_IDX_t, DTYPE_STD_t, DTYPE_UINT_t, DTYPE_UINT
from rgr.constants.types cimport DTYPE_FLOAT


cdef class RegularityConditions:

    cdef:
        readonly:
            double threshold_dev
            PartitionPair pair

    cpdef tuple conditions(self)

    cpdef tuple condition_1(self)
    cpdef tuple condition_2(self)
    cpdef tuple condition_3(self)