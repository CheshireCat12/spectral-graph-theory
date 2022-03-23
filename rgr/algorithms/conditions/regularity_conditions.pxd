
cdef class RegularityConditions:

    cdef:
        int cls_cardinality
        int r, s
        int[::1] classes
        int[::1] r_indices, s_indices
        int[::1] r_degrees, s_degrees
        int[:, ::1] adjacency

        float eps
        float bip_avg_deg
        float bip_edge_density

    cpdef tuple conditions(self)
    cpdef bint condition_1(self, list certificates, list complements)
    cpdef bint condition_2(self, list certificates, list complements)
    cpdef bint condition_3(self, list certificates, list complements)