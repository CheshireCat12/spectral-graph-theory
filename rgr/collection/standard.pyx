import numpy as np

cpdef Graph complete_graph(int n_nodes):
    """
    Complete graph
    All the nodes are linked to every other node

    |V| = n
    |E| = n(n-1)/2
 
    Args:
        n_nodes (int): number of nodes
 
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
        n_nodes (int): number of nodes
 
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
        n_nodes (int): number of nodes

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
           n_nodes (int): number of nodes

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
        n_nodes (int): number of nodes
        prob_edge (float): independent probability that an edge exist between two nodes

    Returns:
        Graph Random Erdos-Renyi graph
        
    """
    shape = (n_nodes, n_nodes)
    rand_adjacency = np.random.choice(np.array([0, 1], dtype=DTYPE_ADJ),
                                 size=shape,
                                 p=[1 - prob_edge, prob_edge])
    low_adjacency = np.tril(rand_adjacency, k=-1)

    return Graph(low_adjacency + low_adjacency.T)
