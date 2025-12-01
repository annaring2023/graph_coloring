"""
Graph coloring module using backtracking algorithm.
Implements 3-coloring (RGB) for undirected graphs.
"""
import networkx as nx
def coloring_algorythm(graph: nx.Graph):
    """
    Color a graph using 3 colors (red, green, blue) via backtracking.
    Constraint: No node can be assigned its original color.

    Args:
        graph (nx.Graph): NetworkX graph to be colored.

    Returns:
        tuple: (True, colored_graph) if successful, (False, None) otherwise.
    """
    for node in graph.nodes():
        graph.nodes[node]['current_color'] = None

    nodes_list = list(graph.nodes())

    def is_safe(node, color):
        """
        Check if a color can be assigned to a node.
        - Cannot be original color
        - Cannot be used by any colored neighbor
        """
        if color == graph.nodes[node]['color']:
            return False

        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor].get('current_color') == color:
                return False
        return True

    def backtrack(index):
        """
        Recursively color nodes using backtracking.
        """
        if index == len(nodes_list):
            return True

        current_node = nodes_list[index]

        for col in ['r', 'g', 'b']:
            if is_safe(current_node, col):
                graph.nodes[current_node]['current_color'] = col

                if backtrack(index + 1):
                    return True

                graph.nodes[current_node]['current_color'] = None
        return False

    if backtrack(0):
        for node in graph.nodes():
            graph.nodes[node]['color'] = graph.nodes[node]['current_color']
            del graph.nodes[node]['current_color']
        return True, graph

    return False, None

if __name__ == "__main__":
    from graph_coloring import read_undirected_graph, draw_colored_graph, is_proper_coloring

    our_graph = read_undirected_graph('right_color.json')
    success, result = coloring_algorythm(our_graph)
    print(f'Success: {success}')
    if success:
        print(f'Is proper coloring: {is_proper_coloring(result)}')
        draw_colored_graph(result)
