from rgr.algorithms.certificates_complements cimport CertificatesComplements
from rgr.algorithms.partition_pair cimport PartitionPair


cdef class RegularityConditions:

    cdef:
        PartitionPair pair

    cpdef tuple conditions(self)

    cpdef tuple condition_1(self)
    cpdef tuple condition_2(self)
    cpdef tuple condition_3(self)