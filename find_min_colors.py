'''Module for find_min_colors function.'''
import networkx as nx
def find_min_colors(graph: nx.Graph) -> int:
    """
    Returns minimum amount of colors required for graph. 
    
    :param graph: Imported graph.
    :type graph: nx.Graph
    :return: Minimum amount of colors.
    :rtype: int
    """
    graph = graph.read_undirected_graph('bigger.json')
    graph = nx.coloring.greedy_color(graph, strategy="random_sequential")

    return len(set(graph.values()))
