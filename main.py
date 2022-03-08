from sgt.graph.graph import Graph
from sgt.graph.star import Star
import numpy as np

def main():
    graph = Star(10)
    print(np.asarray(graph.adjacency))
    print(graph.degree(0))



if __name__ == '__main__':
    main()
