import networkx as nx
import numpy as np

cpdef Graph complete_graph(int n_nodes):
    """
    Complete graph
    All the nodes are linked to every other node

    |V| = n
    |E| = n(n-1)/2
 
    Args:
        n_nodes : int
            number of nodes
 
    Returns: 
        Graph: generated complete graph
 
    """
    shape = (n_nodes, n_nodes)

    # Create the adjacency matrix of the complete graph
    adjacency = np.ones(shape, dtype=DTYPE_ADJ)
    np.fill_diagonal(adjacency, 0)

    return Graph(adjacency)

cpdef Graph cycle_graph(int n_nodes):
    """
    Cycle graph
    The node $n_i$ is linked to the node $n_{i+1}$
    except for node $n_{|V|-1} which is linked to node n_{0}

    |V| = n
    |E| = n
    
    Args:
        n_nodes : int
            number of nodes
 
    Returns:
        Graph: generated cycle graph
    
    Raises:
        ValueError: If the number of nodes n_nodes < 3

    """
    #TODO Handle the exception propagation for cython function returning python object
    if n_nodes < 3:
        raise ValueError(f'A cycle must contain at least 3 nodes')

    shape = (n_nodes - 1, n_nodes - 1)

    adjacency = np.sum([np.diag(np.ones(n_nodes - 1), offset) for offset in [-1, 1]],
                       axis=0,
                       dtype=DTYPE_ADJ)
    adjacency[0, -1] = 1
    adjacency[-1, 0] = 1

    return Graph(adjacency)

cpdef Graph path_graph(int n_nodes):
    """
    Path graph
    The vertices are order $v_0, v_1, ..., v_{n-1}$ 
    such that edges are $(v_{i}, v_{i+1})$ with $i = 0, 1, .., n-2$

    |V| = n
    |E| = n - 1

    Args:
        n_nodes : int
            number of nodes

    Returns:
        Graph: generated path graph

    """
    shape = (n_nodes - 1, n_nodes - 1)

    adjacency = np.sum([np.diag(np.ones(n_nodes - 1), offset) for offset in [-1, 1]],
                       axis=0,
                       dtype=DTYPE_ADJ)

    return Graph(adjacency)

cpdef Graph star_graph(int n_nodes):
    """
    Star graph
    All the nodes $n_{i}$ are linked to node $n_{0}$

    |V| = n
    |E| = n - 1

    Args:
        n_nodes : int
            number of nodes

    Returns:
        Graph generated star graph

       """
    shape = (n_nodes, n_nodes)

    adjacency = np.zeros(shape, dtype=DTYPE_ADJ)

    # connect all the nodes to node 0
    adjacency[0, 1:] = 1
    adjacency[1:, 0] = 1

    return Graph(adjacency)

cpdef Graph erdos_renyi_graph(int n_nodes, float prob_edge):
    """
    Erdos-Renyi graph
    The nodes are randomly connected
    (i.e., each edge is included to the graph with probability ´prob_edge´
    
    Args:
        n_nodes : int
            number of nodes
        prob_edge : float 
            independent probability that an edge exist between two nodes

    Returns:
        Graph Random Erdos-Renyi graph
        
    """
    shape = (n_nodes, n_nodes)
    rand_adjacency = np.random.choice(np.array([0, 1], dtype=DTYPE_ADJ),
                                      size=shape,
                                      p=[1 - prob_edge, prob_edge])
    low_adjacency = np.tril(rand_adjacency, k=-1)

    return Graph(low_adjacency + low_adjacency.T)

cpdef Graph peterson_graph():
    """
    Create the perterson graph
    
    Returns:
        Graph Peterson graph
    """
    adjacency = np.array(
        [[0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
         [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 1, 0, 0]],
        dtype=DTYPE_ADJ)

    return Graph(adjacency)

cpdef Graph stochastic_block_model(int n_nodes,
                                   int n_clusters,
                                   double intra_noise,
                                   double inter_noise,
                                   int seed=42):
    """
    Create a graph with partition blocks.
    The ´intra_noise´ and ´inter_noise´ parameters model the percentage
    of noise in and between the blocks, respectively
    
    Args:
        n_nodes: int
            Number of nodes
        n_clusters: int
            Number of blocks
        intra_noise: float
            Probability of noise in the blocks
        inter_noise: float
            Probability of noise between blocks
        seed: int

    Returns:
        Graph
    """
    cdef:
        list sizes
        np.ndarray probs, adjacency

    sizes = [n_nodes // n_clusters] * n_clusters
    sizes[-1] += n_nodes - sum(sizes)

    probs = np.ones((n_clusters, n_clusters)) * inter_noise
    np.fill_diagonal(probs, 1 - intra_noise)

    block_graph = nx.stochastic_block_model(sizes, probs, seed=0)

    return Graph(nx.to_numpy_array(block_graph).astype(DTYPE_ADJ))

# cpdef Graph block_model(int n_nodes, np.ndarray dims, internoise_lvl, internoise_val, intranoise_lvl, intranoise_val):
# np.random.seed(0)
# G = np.tril(np.random.random((n_nodes, n_nodes)).astype('float32') < internoise_lvl, -1).astype('int8')
# G = np.multiply(G, internoise_val)
# print(G)
# GT = np.tril(np.zeros((n_nodes, n_nodes), dtype=DTYPE_ADJ), -1)
# print(GT)
#
# x = 0
# for dim in dims:
#     cluster = np.tril(np.ones((dim, dim), dtype="float32"), -1)
#     mask = np.tril(np.random.random((dim, dim)).astype("float32") < intranoise_lvl, -1)
#
#     if intranoise_val== 0:
#         cluster += mask
#         indices = (cluster == 2)
#         cluster[indices] = 0
#     else:
#         mask = np.multiply(mask, intranoise_val)
#         cluster += mask
#         indices = (cluster > 1)
#         cluster[indices] = intranoise_val
#
#     G[x:x + dim, x:x + dim] = cluster.astype('int8')
#     GT[x:x + dim, x:x + dim] = np.tril(np.ones(dim, dtype="int8"), -1)
#
#     x += dim
#
# # print(G + G.T)
# # print(GT+GT.T)
# return Graph(G + G.T)
