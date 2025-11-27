"""
Graph coloring module using backtracking algorithm.
Implements 3-coloring (RGB) for undirected graphs.
"""
import networkx as nx

colors = ['r', 'g', 'b']
def coloring_algorythm(graph: nx.Graph):
    """
    Color a graph using 3 colors (red, green, blue) via backtracking.
    
    Args:
        graph (nx.Graph): NetworkX graph to be colored.
    
    Returns:
        tuple: (True, colored_graph) if successful, (False, None) otherwise.
    """
    for node in graph.nodes():
        graph.nodes[node]['color'] = None
    nodes_list = list(graph.nodes())

    def is_safe(node, color):
        """
        Check if a color can be assigned to a node.
        
        Args:
            node: Node to check.
            color (str): Color to test ('r', 'g', or 'b').
        
        Returns:
            bool: True if color is safe, False otherwise.
        """
        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor]['color'] == color:
                return False
        return True
        
    def backtrack(index):
        """
        Recursively color nodes using backtracking.
        
        Args:
            index (int): Current node index in nodes_list.
        
        Returns:
            bool: True if coloring successful, False otherwise.
        """
        if index == len(nodes_list):
            return True
        
        current_node = nodes_list[index]

        
        for col in colors:
            if is_safe(current_node, col):
                graph.nodes[current_node]['color'] = col
                
                if backtrack(index + 1):
                    return True
                graph.nodes[current_node]['color'] = None
        return False
    if backtrack(0):
        return True, graph
    return False, None

if __name__ == "__main__":
    from graph_coloring import read_undirected_graph, draw_colored_graph, is_proper_coloring
    
    graph = read_undirected_graph('wrong_color.json')
    success, result = coloring_algorythm(graph)
    print(f'Succes: {success}')
    if success:
        print(f'If its right: {is_proper_coloring(result)}')
        draw_colored_graph(result)
