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


if __name__ == "__main__":
    from graph_coloring import read_undirected_graph, draw_colored_graph, is_proper_coloring

    our_graph = read_undirected_graph('right_color.json')
    success, result = coloring_algorythm(our_graph)
    print(f'Success: {success}')
    if success:
        print(f'Is proper coloring: {is_proper_coloring(result)}')
        #draw_colored_graph(our_graph)
        draw_colored_graph(result)
