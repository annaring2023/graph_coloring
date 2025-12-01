"""
Graph coloring module using backtracking algorithm.
Implements 3-coloring (RGB) for undirected graphs.
"""
import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt

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
    NodeDataView({1: {'color': 'b'}, 2: {'color': 'g'}, 3: {'color': 'g'}, 4: {'color': 'r'}})
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

def is_proper_coloring(graph):
    """
    Check whether the graph coloring is proper.

    A coloring is proper if no two adjacent nodes share the same color.
    The color of each node is stored in the node attribute "color".

    Args:
        graph (nx.Graph): Graph whose coloring should be checked.

    Returns:
        bool: True if the coloring is proper, False otherwise.
    """
    for u, v in graph.edges():
        if graph.nodes[u]["color"] == graph.nodes[v]["color"]:
            return False
    return True

def graph_to_dict(graph):
    """
    Convert a graph into a dictionary sorted by node degree.

    Nodes are sorted in descending order of their degree (number of neighbors).
    For each node, the function stores its color and the set of its neighbors.

    Args:
        graph (nx.Graph): Graph to convert.

    Returns:
        dict: Dictionary of the form
            {
                node: {
                    "color": <node color>,
                    "neighbors": {neighbor1, neighbor2, ...}
                },
                ...
            }
    """
    sorted_nodes = sorted(graph.nodes(), key=lambda v: graph.degree[v], reverse=True)
    result = {}
    for node in sorted_nodes:
        color = graph.nodes[node].get("color")
        neighbors = sorted(graph.neighbors(node))
        result[node] = {"color": color, "neighbors": neighbors}
    return result

def draw_colored_graph(graph: nx.Graph) -> None:
    """
    Draw an undirected graph, using node colors from the "color" attribute.

    Node colors are taken from graph.nodes[node]["color"]. One-letter color
    codes such as 'r', 'g', 'b', etc. are supported. If a node has no color,
    the default color "gray" is used.

    The function computes node positions with a spring layout and then
    renders the graph using matplotlib.

    Args:
        graph (nx.Graph): Graph to visualize.

    Returns:
        None
    """
    pos = nx.spring_layout(graph, seed=42)
    node_colors = [graph.nodes[n].get("color", "gray") for n in graph.nodes()]
    plt.figure(figsize=(5, 5))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="black",
        node_size=800,
        font_size=10,
        font_color="white",
        linewidths=1
    )
    plt.title("Colored graph")
    plt.axis("off")
    plt.show()



def coloring_algorythm(graph: nx.Graph):
    """
    Color a graph using 3 colors (red, green, blue) via backtracking.
    Constraint: No node can be assigned its original color.

    Args:
        graph (nx.Graph): NetworkX graph to be colored.

    Returns:
        tuple: (True, colored_graph) if successful, (False, None) otherwise.
    """
    working_graph = graph.copy()
    nodes_list = list(working_graph.nodes())

    for node in working_graph.nodes():
        working_graph.nodes[node]['color'] = None

    def is_safe(node, color):
        if color == graph.nodes[node]['color']:
            return False
        for neighbor in working_graph.neighbors(node):
            if working_graph.nodes[neighbor]['color'] == color:
                return False
        return True

    def backtrack(index):
        if index == len(nodes_list):
            return True
        current_node = nodes_list[index]
        for col in ['r', 'g', 'b']:
            if is_safe(current_node, col):
                working_graph.nodes[current_node]['color'] = col
                if backtrack(index + 1):
                    return True
                working_graph.nodes[current_node]['color'] = None
        return False

    if backtrack(0):
        return True, working_graph
    return False, None


def main():
    '''
    Entry point for the command-line interface.

    This function parses command-line arguments, reads the graph from the
    provided JSON file and performs one of the following actions:

    - "check": check whether the graph coloring is proper and print the result.
    - "dict":  print a dictionary representation of the graph.
    - "draw":  display a colored visualization of the graph.

    Command-line usage:
        python graph_coloring.py filepath [--mode {check,dict,draw}]

    Examples:
        python graph_coloring.py graph_v1.json
        python graph_coloring.py graph_v1.json --mode dict
        python graph_coloring.py graph_v1.json --mode draw
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath",)
    parser.add_argument("--mode", choices=["check", "dict", "draw", "color"],default="draw")
    args = parser.parse_args()
    graph = read_undirected_graph(args.filepath)
    if args.mode == "check":
        print("Proper coloring:", is_proper_coloring(graph))
    elif args.mode == "dict":
        print(graph_to_dict(graph))
    elif args.mode == "draw":
        draw_colored_graph(graph)
    elif args.mode == "color":
        success, result = coloring_algorythm(graph)
        print(f'Success: {success}')
        if success:
            print(f'Is proper coloring: {is_proper_coloring(result)}')
            draw_colored_graph(result)

if __name__ == "__main__":
    main()
