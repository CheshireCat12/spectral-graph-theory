from sgt.graph import Graph

def main():
    graph = Graph(20)
    print(graph.adjacency.base)

    print(graph.degree(4))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

