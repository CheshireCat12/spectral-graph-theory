# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

np.import_array()

cdef class PartitionPair:
    """Data class used to handle information between a pair of partitions."""

    def __init__(self,
                 np.ndarray[DTYPE_ADJ_t, ndim=2] adjacency,
                 list partitions,
                 int r,
                 int s,
                 double eps):
        """

        Args:
            adjacency: np.ndarray[DTYPE_ADJ_t, ndim=2]
            partitions: List[np.ndarray[DTYPE_IDX_t, ndim=1]
            r: int
            s: int
            eps: double
        """
        self.adjacency = adjacency
        self.r = r
        self.s = s
        self.eps = eps

        self.r_indices = partitions[r]
        self.s_indices = partitions[s]

        # Bipartite adjacency matrix
        self.bip_adj = self.adjacency[np.ix_(self.s_indices, self.r_indices)]
        # temp = self.adjacency[np.ix_(self.s_indices, self.r_indices)]

        # Cardinality of the partitions
        self.prts_size = len(self.s_indices) #self.bip_adj.shape[0]

        self.bip_sum_edges = np.sum(self.bip_adj)

        # Bipartite average degree
        # To have a faster summation of the bipartite degrees
        # I directly sum the elements over the whole matrix,
        # so I don't have to divide the sum by 2
        self.bip_avg_deg = self.bip_sum_edges / self.prts_size

        self.bip_density = self.bip_sum_edges / (self.prts_size**2)
        # print(f'bip sum edges: {self.bip_sum_edges}')
        # print(f'prts_size: {self.prts_size}')

        self.s_degrees = np.sum(self.bip_adj, axis=1).astype(DTYPE_UINT)
        self.r_degrees = np.sum(self.bip_adj, axis=0).astype(DTYPE_UINT)


cdef class PartitionPairFast:
    """Data class used to handle information between a pair of partitions."""

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    def __init__(self,
                 # np.ndarray[DTYPE_ADJ_t, ndim=2] adjacency,
                 DTYPE_ADJ_t[:, ::1] adjacency,
                 list partitions,
                 int r,
                 int s,
                 double eps):
        """

        Args:
            adjacency: np.ndarray[DTYPE_ADJ_t, ndim=2]
            partitions: List[np.ndarray[DTYPE_IDX_t, ndim=1]
            r: int
            s: int
            eps: double
        """
        cdef:
            Py_ssize_t i, j
            Py_ssize_t max_i, max_j
        # self.adjacency = adjacency
        self.adjacency_ = adjacency
        self.r = r
        self.s = s
        self.eps = eps

        # self.r_indices = partitions[r]
        # self.s_indices = partitions[s]

        self.r_indices_ = partitions[r]
        self.s_indices_ = partitions[s]

        # # Bipartite adjacency matrix

        # self.bip_adj = np.array(self.adjacency_)[np.ix_(self.s_indices_, self.r_indices_)]
        # print(np.array(self.r_indices_))
        #
        max_i = self.s_indices_.shape[0]
        max_j = self.r_indices_.shape[0]
        self.bip_adj_ = np.empty((max_i, max_j), dtype = DTYPE_ADJ)
        self.bip_adj_ = np.zeros((max_i, max_j), dtype = DTYPE_ADJ)
        for i in range(max_i):
            for j in range(max_j):
                self.bip_adj_[i][j] = self.adjacency_[self.s_indices_[i]][self.r_indices_[j]]

        # Cardinality of the partitions
        # self.prts_size = len(self.s_indices) #self.bip_adj.shape[0]
        # self.prts_size = self.adjacency_.shape[0]
        self.prts_size = adjacency.shape[0]

        self.bip_sum_edges = c_sum_matrix(self.bip_adj)
        #
        # # Bipartite average degree
        # # To have a faster summation of the bipartite degrees
        # # I directly sum the elements over the whole matrix,
        # # so I don't have to divide the sum by 2
        self.bip_avg_deg = self.bip_sum_edges / self.prts_size
        #
        self.bip_density = self.bip_sum_edges / (self.prts_size**2)
        # # print(f'bip sum edges: {self.bip_sum_edges}')
        # # print(f'prts_size: {self.prts_size}')
        #
        # # self.s_degrees = np.sum(self.bip_adj, axis=1).astype(DTYPE_UINT)
        self.s_degrees = c_sum_mat_axis(self.bip_adj, axis=1)
        # self.r_degrees = c_sum_mat_axis(self.bip_adj, axis=0)
        self.r_degrees = np.sum(self.bip_adj, axis=0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long c_sum_matrix(DTYPE_ADJ_t[:, ::1] arr1):
    cdef Py_ssize_t x_max = arr1.shape[0]
    cdef Py_ssize_t y_max = arr1.shape[1]
    cdef long results = 0
    cdef long x, y

    for x in range(x_max):
    # for x in prange(x_max, nogil=True):
        for y in range(y_max):
            results += arr1[x, y]

    return results

def sum_matrix(arr):
    return c_sum_matrix(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_UINT_t, ndim=1] c_sum_mat_axis(DTYPE_ADJ_t[:, ::1] arr1, int axis):
    cdef Py_ssize_t x_max = arr1.shape[0]
    cdef Py_ssize_t y_max = arr1.shape[1]
    cdef Py_ssize_t  x, y
    cdef DTYPE_ADJ_t[:, :] arr1_T
    cdef np.ndarray[DTYPE_UINT_t, ndim=1] results

    if axis == 0:
        arr1_T = arr1.T
        x_max = arr1_T.shape[0]
        y_max = arr1_T.shape[1]

    results = np.zeros(x_max, dtype=DTYPE_UINT)

    if axis == 1:
        for x in prange(x_max, nogil=True):
            for y in range(y_max):
                results[x] += arr1[x, y]
    else:
        for y in prange(y_max, nogil=True):
            for x in range(x_max):
                results[x] += arr1_T[x, y]

    return results

def sum_mat_axis(arr, axis):
    return c_sum_mat_axis(arr, axis)

