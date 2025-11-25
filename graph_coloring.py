import json
import networkx as nx


def read_undirected_graph(filepath: str) -> nx.Graph:
    """
    Read an undirected graph from a JSON file and construct a NetworkX Graph.

    The JSON file must contain a dictionary where each key is a string representing
    a node ID, and each value is an object with two fields:
        - "color": a string specifying the node's color
        - "edge_with": a list of integers representing adjacent nodes
    JSON Example:
        {
            "1": {
                "color": "b",
                "edge_with": [2, 3, 4]
            },
            "2": {
                "color": "b",
                "edge_with": [1, 4]
            },
            "3": {
                "color": "g",
                "edge_with": [1]
            },
            "4": {
                "color": "r",
                "edge_with": [1, 2]
            }
        }

    Each edge is added only once (u < v) because the graph is undirected.

    Args:
        filepath (str): Path to the JSON file containing graph data.

    Returns:
        nx.Graph: A NetworkX undirected graph with node colors and edges.

    Example:
    >>> graph = read_undirected_graph('graph_v1.json')
    >>> graph.nodes
    NodeView((1, 2, 3, 4))
    >>> graph.edges
    EdgeView([(1, 2), (1, 3), (1, 4), (2, 4)])
    >>> graph.nodes(data=True)
    NodeDataView({1: {'color': 'b'}, 2: {'color': 'b'}, 3: {'color': 'g'}, 4: {'color': 'r'}})
    """
    with open(filepath, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    graph_data = {int(node): info for node, info in raw_data.items()}

    graph = nx.Graph()

    # Додаємо вершини з атрибутом color
    for node, info in graph_data.items():
        graph.add_node(node, color=info["color"])


    # Додаємо ребра
    for node, info in graph_data.items():
        for adjacent_node in info["edge_with"]:
            if node < adjacent_node:   # щоб не додавати (2,1), якщо вже було (1,2)
                graph.add_edge(node, adjacent_node)

    return graph

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
