import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

from rgr.graph.graph cimport Graph
from rgr.constants.types cimport DTYPE_ADJ


cpdef int[::1] random_partition_init(int n_nodes, int n_classes):
    """
    Create the initial random partitioning of Alon algorithm.
    The nodes in graph G are partitioned into k classes.
    Where each class C_k has equal number of 
    
    Args:
        n_classes: 
        n_nodes: 

    Returns:

    """
    # TODO: rename nodes_per_cls by nodes_per_cls or even cls_cardinality
    cls_cardinality = n_nodes // n_classes

    classes = np.repeat(range(1, n_classes+1),
                        cls_cardinality).astype(DTYPE_ADJ)
    np.random.shuffle(classes)

    return classes

cpdef alon_condition_1(int bip_avg_deg, int cls_cardinality, float eps):
    return bip_avg_deg < (eps**3) * cls_cardinality

cpdef alon_condition_2(adjacency, s_indices, r_indices, s_r_degrees, bip_avg_deg, cls_cardinality, eps):
    threshold_deviation = (eps**4) * cls_cardinality
    threshold_n_nodes = (1/8) * (eps**4) * cls_cardinality

    s_degrees = np.asarray(s_r_degrees)[s_indices]

    is_irregular = False
    # s, r certificates
    certificates = [[], []]
    # s, r complements
    complements = [[], []]

    deviated_nodes_mask = np.abs(s_degrees-bip_avg_deg) >= threshold_deviation
    # print(f'threshold dev {threshold_deviation}')
    # print(f'avg deg {bip_avg_deg}')
    #
    # print(f'degrees {s_degrees}')
    # print(f'deviated nodes {deviated_nodes_mask}')

    if np.sum(deviated_nodes_mask) > threshold_n_nodes:
        is_irregular = True

        s_certs = s_indices[deviated_nodes_mask]
        s_complements = np.setdiff1d(s_indices, s_certs)

        b_mask = np.asarray(adjacency)[np.ix_(s_certs, r_indices)] > 0
        b_mask = b_mask.any(0)

        r_certs = r_indices[b_mask]
        r_compls = np.setdiff1d(r_indices, r_certs)

        certificates = [s_certs.tolist(), r_certs.tolist()]
        complements = [s_complements.tolist(), r_compls.tolist()]

    return is_irregular, certificates, complements

cpdef find_Yp(s_degrees, s_indices, bip_avg_deg, cls_cardinality, eps):
    threshold_deviation = (eps ** 4) * cls_cardinality
    mask = np.abs(s_degrees - bip_avg_deg) >= threshold_deviation
    yp_i = np.where(mask == True)[0]

    return yp_i

cpdef compute_y0(ngh_dev, s_indices, yp_i, cls_cardinality, eps):
    threshold_dev = 2 * eps**4 * cls_cardinality
    rect_mat = ngh_dev[yp_i]

    boolean_mat = rect_mat > threshold_dev
    cardinality_by0s = np.sum(boolean_mat, axis=1)

    y0_idx = np.argmax(cardinality_by0s)
    aux = yp_i[y0_idx]

    y0 = s_indices[aux]

    if cardinality_by0s[y0_idx] > (eps ** 4 * cls_cardinality / 4.0):
        cert_s = s_indices[boolean_mat[y0_idx]]
        return cert_s, y0
    else:
        return None, y0




cpdef alon_condition_3(adjacency, s_indices, r_indices, s_r_degrees, bip_adj, bip_avg_deg, cls_cardinality, eps):
    is_irregular = False
    # s, r certificates
    certificates = [[], []]
    # s, r complements
    complements = [[], []]

    ngh_dev = neighbourhood_deviation(bip_adj, bip_avg_deg, cls_cardinality)

    s_degrees = np.asarray(s_r_degrees)[s_indices]

    yp_filter = find_Yp(s_degrees, s_indices, bip_avg_deg, cls_cardinality, eps)

    if np.asarray(yp_filter).size == 0:
        is_irregular = True
        return is_irregular, certificates, complements

    s_certs, y0 = compute_y0(ngh_dev, s_indices, yp_filter, cls_cardinality, eps)

    if s_certs is None:
        is_irregular = False
        return is_irregular, certificates, complements
    else:
        assert np.array_equal(np.intersect1d(s_certs, s_indices), s_certs) == True, "cert_is not subset of s_indices"
        assert (y0 in s_indices) == True, "y0 not in s_indices"

        is_irregular = True
        b_mask = np.asarray(adjacency)[np.ix_(np.array([y0]), r_indices)] > 0
        r_certs = r_indices[b_mask[0]]
        assert np.array_equal(np.intersect1d(r_certs, r_indices), r_certs) == True, "cert_is not subset of s_indices"

        # [BUG] cannot do set(s_indices) - set(s_certs)
        s_compls = np.setdiff1d(s_indices, s_certs)
        r_compls = np.setdiff1d(r_indices, r_certs)
        assert s_compls.size + s_certs.size == cls_cardinality, "Wrong cardinality"
        assert r_compls.size + r_certs.size == cls_cardinality, "Wrong cardinality"

        return is_irregular, [r_certs.tolist(), s_certs.tolist()], [r_compls.tolist(), s_compls.tolist()]

cpdef neighbourhood_deviation(bip_adj, bip_avg_deg, cls_cardinality):
    mat = np.matmul(np.asarray(bip_adj.T), np.asarray(bip_adj))
    print(mat)
    mat = mat - (bip_avg_deg**2) / cls_cardinality

    return mat


cpdef classes_pair(int[:, ::1] adjacency, int[::1] classes, int r, int s, eps):
    """
    
    Args:
        adjacency: 
        classes: 
        r: 
        s: 
        eps: 

    Returns:

    """
    s_indices = np.where(np.asarray(classes)==s)[0]
    r_indices = np.where(np.asarray(classes)==r)[0]

    # Bipartite adjacency matrix
    bip_adj = np.asarray(adjacency)[np.ix_(s_indices, r_indices)]

    # Cardinality of the classes
    cls_cardinality = bip_adj.shape[0]

    bip_sum_edges = np.sum(bip_adj)

    # Bipartite average degree
    # To have a faster summation of the bipartite degrees
    # I directly sum the elements over the whole matrix,
    # so I don't have to divide the sum by 2
    bip_avg_deg = bip_sum_edges / cls_cardinality
    # np.sum(np.sum(bip_adj, axis=0) + np.sum(bip_adj, axis=1)) / (2 * cls_cardinality)

    # Bipartite edge density
    bip_edge_density = bip_sum_edges / (cls_cardinality**2)

    s_r_degrees = np.zeros(classes.shape[0], dtype=DTYPE_ADJ)
    s_r_degrees[s_indices] = np.sum(bip_adj, axis=1)
    s_r_degrees[r_indices] = np.sum(bip_adj, axis=0)

    print(alon_condition_1(bip_avg_deg, cls_cardinality=cls_cardinality, eps=eps))
    print(alon_condition_2(adjacency, s_indices, r_indices, s_r_degrees, bip_avg_deg, cls_cardinality, eps))
    print(alon_condition_3(adjacency, s_indices, r_indices, s_r_degrees, bip_adj, bip_avg_deg, cls_cardinality, eps))
    neighbourhood_deviation(bip_adj, bip_avg_deg, cls_cardinality)

@boundscheck(False)
@wraparound(False)
cpdef void degrees(Graph graph):

    np.sum(graph.adjacency, axis=0)
    # np.asarray(graph.adjacency) @ np.asarray(graph.adjacency)

# @boundscheck(False)
# @wraparound(False)
cpdef void degrees2(Graph graph):
    cdef:
        int i, j
        int[::1] degs

    m, n = graph.adjacency.shape
    degs = np.zeros(m, dtype=np.int32)

    for i in range(m):
        for j in range(n):
            print(i, j)
            degs[i] += graph.adjacency[i][j]

