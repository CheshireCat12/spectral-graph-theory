import numpy as np

np.import_array()

cpdef matrix_reduction(np.ndarray[DTYPE_ADJ_t, ndim=2] adjacency,
                       int n_partitions,
                       list partitions):
    cdef:
        int r, s
        np.ndarray[np.float32_t, ndim=2] reduced_adjacency

    reduced_adjacency = np.zeros((n_partitions, n_partitions),
                                 dtype=np.float32)

    for r in range(2, n_partitions+1):
        r_indices = partitions[r]
        for s in range(1, r):
            s_indices = partitions[s]

            bip_adj_mat = adjacency[np.ix_(s_indices, r_indices)]
            classes_n = bip_adj_mat.shape[0]
            bip_dens = np.sum(bip_adj_mat) / (classes_n**2.)
            # print(f'r: {r - 1}, s: {s - 1}, bip dens: {bip_dens}')
            reduced_adjacency[r-1, s-1] = reduced_adjacency[s-1, r-1] = bip_dens

    return reduced_adjacency