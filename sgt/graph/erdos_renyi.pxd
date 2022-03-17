from sgt.graph.graph cimport Graph

cdef class ErdosRenyi(Graph):

    cdef:
        double p

    cpdef void __init_graph(self)