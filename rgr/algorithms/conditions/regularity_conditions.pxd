from rgr.algorithms.partition_pair cimport PartitionPair


cdef class RegularityConditions:

    cdef:
        PartitionPair pair

    # @staticmethod
    # cpdef tuple conditions(self, PartitionPair pair)

    cpdef bint condition_1(self, list certificates, list complements)
    cpdef bint condition_2(self, list certificates, list complements)
    cpdef bint condition_3(self, list certificates, list complements)