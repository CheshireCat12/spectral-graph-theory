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
                 # np.ndarray[DTYPE_ADJ_t, ndim=2, mode='c'] adjacency,
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
            Py_ssize_t idx_s, idx_r
            Py_ssize_t max_i, max_j
            DTYPE_UINT_t[::1] r_indices, s_indices
            DTYPE_ADJ_t[:, ::1] bip_adj
        self.adjacency = adjacency
        self.r = r
        self.s = s
        self.eps = eps

        r_indices = partitions[r]
        s_indices = partitions[s]

        # # Bipartite adjacency matrix

        max_i = s_indices.shape[0]
        max_j = r_indices.shape[0]
        bip_adj = np.zeros((max_i, max_j), dtype = DTYPE_ADJ)
        # for i in range(max_i):
        for i in prange(max_i, nogil=True):
            for j in range(max_j):
                idx_s = s_indices[i]
                idx_r = r_indices[j]
                bip_adj[i][j] = adjacency[idx_s][idx_r]

        # Post initialization of class variables to avoid
        self.r_indices = r_indices
        self.s_indices = s_indices
        self.bip_adj = bip_adj

        # Cardinality of the partitions
        self.prts_size = bip_adj.shape[0]

        self.bip_sum_edges = c_sum_matrix(bip_adj)

        # assert self.prts_size != 0, f'Partition size is zero!'

        # Bipartite average degree
        # To have a faster summation of the bipartite degrees
        # I directly sum the elements over the whole matrix,
        # so I don't have to divide the sum by an extra 2
        self.bip_avg_deg = self.bip_sum_edges / float(self.prts_size)

        self.bip_density = self.bip_sum_edges / float(self.prts_size**2)

        self.s_degrees = c_sum_mat_axis(bip_adj, axis=1)
        self.r_degrees = c_sum_mat_axis(bip_adj, axis=0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long c_sum_matrix(DTYPE_ADJ_t[:, ::1] arr1):
    cdef:
        Py_ssize_t x, y
        Py_ssize_t x_max = arr1.shape[0]
        Py_ssize_t y_max = arr1.shape[1]
        long results = 0

    for x in range(x_max):
    # for x in prange(x_max, nogil=True):
        for y in range(y_max):
            results += arr1[x, y]

    return results

def sum_matrix(arr):
    return c_sum_matrix(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_UINT_t[::1] c_sum_mat_axis(DTYPE_ADJ_t[:, ::1] arr1, int axis):
    cdef:
        Py_ssize_t row, col
        Py_ssize_t x_max = arr1.shape[0]
        Py_ssize_t y_max = arr1.shape[1]
        DTYPE_UINT_t[::1] results


    if axis == 1:
        results = np.zeros(x_max, dtype=DTYPE_UINT)
    else:
        results = np.zeros(y_max, dtype=DTYPE_UINT)

    if axis == 1:
        # for row in range(x_max):
        for row in prange(x_max, nogil=True):
            for col in range(y_max):
                results[row] += arr1[row, col]
    else:
        # for col in range(y_max):
        for col in prange(y_max, nogil=True):
            for row in range(x_max):
                results[col] += arr1[row, col]

    return results

def sum_mat_axis(arr, axis):
    return c_sum_mat_axis(arr, axis)

