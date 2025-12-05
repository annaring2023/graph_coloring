"""
Graph coloring module using backtracking algorithm.
Implements 3-coloring (RGB) for undirected graphs.
"""
import time
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
    plt.figure(figsize=(8, 6))
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
    nodes_list = sorted(working_graph.nodes(), key=lambda v: graph.degree[v], reverse=True)

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

def perfomance(graph: nx.Graph):
    """
    Docstring for testing perfomance between built-in \
algorithm and NetworkX's Largest_First algorigthm.
    
    :param graph: NetworkX graph to be tested.
    :type graph: nx.Graph()
    """
    start_time = time.perf_counter()
    coloring_algorythm(graph)
    middle_time = time.perf_counter()
    nx.greedy_color(graph, strategy="largest_first")
    end_time = time.perf_counter()
    builtin_time = middle_time - start_time
    nx_time = end_time - middle_time
    print(f"Вбудований Алгоритм: {builtin_time:.5f}")
    print(f"NetworkX   Алгоритм: {nx_time:.5f}")
    print("\n" + "="*60)
    if builtin_time < nx_time:
        print(f"Вбудований Алгоритм ліпший за NetworkX в {nx_time/builtin_time:.5f} рази.")
    if nx_time < builtin_time:
        print(f"NetworkX Алгоритм ліпший за Вбудований на {builtin_time/nx_time:.5f} рази.")


def main():
    '''
    Entry point for the command-line interface.

    This function parses command-line arguments, reads the graph from the
    provided JSON file and performs one of the following actions:

    - "check": check whether the graph coloring is proper and print the result.
    - "dict":  print a dictionary representation of the graph.
    - "draw":  display a colored visualization of the graph.
    - "color": color the graph using backtracking algorithm.
    - "info":  show general information about the graph.

    Command-line usage:
        python FULL_CODE.py filepath [--mode {check,dict,draw,color,info}] [--verbose]

    Examples:
        python FULL_CODE.py graph.json Показати інформацію про граф (за замовчуванням)
        python FULL_CODE.py graph.json --mode check  Перевірити правильність розфарбування
        python FULL_CODE.py graph.json --mode color Розфарбувати граф алгоритмом backtracking
        python FULL_CODE.py graph.json --mode dict Показати граф у вигляді словника
        python FULL_CODE.py graph.json --mode draw Візуалізувати граф
        python FULL_CODE.py graph.json --mode info --verbose Показати детальну інформацію про граф
    '''
    # Інтерфейс: створення парсера аргументів командного рядка
    parser = argparse.ArgumentParser(description="Програма для роботи з розфарбуванням графів")
    # Інтерфейс: обов'язковий аргумент - шлях до файлу
    parser.add_argument("filepath", type=str, help="Шлях до JSON файлу з даними графа")
    # Інтерфейс: режим роботи програми
    parser.add_argument("--mode", "-m",
        choices=["check", "dict", "draw", "color", "info", "perf"],
        default="info",
        help="""Режим роботи програми:
  check  - перевірити правильність розфарбування графа
  dict   - вивести граф у вигляді словника, відсортованого за степенем вершин
  draw   - відобразити візуалізацію графа
  color  - розфарбувати граф алгоритмом backtracking (3 кольори: r, g, b)
  info   - показати загальну інформацію про граф (за замовчуванням)
  perf   - порівняти ефективність між вбудованим алгоритмом та алгоритмом networkx""")
    # Інтерфейс: додатковий аргумент для детального виводу
    parser.add_argument("--verbose", "-v", action="store_true",\
 help="Показати детальну інформацію про виконання")
    # Інтерфейс: опція для збереження результату
    parser.add_argument("--output", "-o", type=str,\
 help="Зберегти результат розфарбування у файл (тільки для режиму color)")
    args = parser.parse_args()

    try:
        graph = read_undirected_graph(args.filepath)
        # Інтерфейс: виконання відповідної дії залежно від обраного режиму
        if args.mode == "check":
            # Інтерфейс: режим перевірки розфарбування
            print("\n" + "="*60)
            print("ПЕРЕВІРКА РОЗФАРБУВАННЯ ГРАФА")
            print("="*60)
            is_proper = is_proper_coloring(graph)
            if is_proper:
                print("✓ Розфарбування є ПРАВИЛЬНИМ")
                print("  Жодні дві сусідні вершини не мають однакового кольору.")
            else:
                print("✗ Розфарбування є НЕПРАВИЛЬНИМ")
                print("  Знайдено сусідні вершини з однаковим кольором:")
                for u, v in graph.edges():
                    if graph.nodes[u]["color"] == graph.nodes[v]["color"]:
                        print(f"    Вершина {u} (колір: {graph.nodes[u]\
['color']}) та вершина {v} (колір: {graph.nodes[v]['color']})")
            print("="*60 + "\n")

        elif args.mode == "dict":
            # Інтерфейс: режим виводу графа у вигляді словника
            print("\n" + "="*60)
            print("СЛОВНИКОВЕ ПРЕДСТАВЛЕННЯ ГРАФА")
            print("(відсортовано за степенем вершин)")
            print("="*60)
            graph_dict = graph_to_dict(graph)
            print(graph_dict)
            print("="*60 + "\n")

        elif args.mode == "draw":
            # Інтерфейс: режим візуалізації графа
            print("\n" + "="*60)
            print("ВІЗУАЛІЗАЦІЯ ГРАФА")
            draw_colored_graph(graph)
            print("="*60 + "\n")

        elif args.mode == "color":
            # Інтерфейс: режим розфарбування графа
            print("\n" + "="*60)
            print("РОЗФАРБУВАННЯ ГРАФА АЛГОРИТМОМ BACKTRACKING")
            print("="*60)

            success, result = coloring_algorythm(graph)

            if success:
                print("✓ Розфарбування успішно знайдено!")
                print(f"✓ Перевірка правильності: {is_proper_coloring(result)}")

                print("\nРозподіл кольорів:")
                color_count = {'r': 0, 'g': 0, 'b': 0}
                for node in result.nodes():
                    color = result.nodes[node]['color']
                    color_count[color] = color_count.get(color, 0) + 1

                color_names = {'r': 'червоний', 'g': 'зелений', 'b': 'синій'}
                for color, count in color_count.items():
                    print(f"  {color_names[color]} ({color}): {count} вершин")

                print("\nРозфарбування вершин:")
                for node in sorted(result.nodes()):
                    color = result.nodes[node]['color']
                    neighbors = list(result.neighbors(node))
                    print(f"  Вершина {node}: колір '{color}'\
 ({color_names[color]}), сусідні: {neighbors}")
                draw_colored_graph(result)
                # Інтерфейс: збереження результату, якщо вказано параметр --output
                if args.output:
                    output_data = {}
                    for node in result.nodes():
                        output_data[node] = {
                            "color": result.nodes[node]['color'],
                            "edge_with": list(result.neighbors(node))
                        }
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2)
                    print(f"\n✓ Результат збережено у файл: {args.output}")
            else:
                print("✗ Не вдалося знайти правильне розфарбування")
                print("  Граф не може бути розфарбований 3-ома кольорами")
            print("="*60 + "\n")

        elif args.mode == "info":
            # Інтерфейс: режим виводу загальної інформації про граф
            print("\n" + "="*60)
            print("ІНФОРМАЦІЯ ПРО ГРАФ")
            print("="*60)
            print(f"Кількість вершин: {len(graph.nodes())}")
            print(f"Кількість ребер: {len(graph.edges())}")

            if len(graph.nodes()) > 0:
                degrees = [graph.degree(node) for node in graph.nodes()]
                print(f"Мінімальний степінь: {min(degrees)}")
                print(f"Максимальний степінь: {max(degrees)}")

            is_proper = is_proper_coloring(graph)
            if is_proper:
                print("\nПравильність розфарбування: ✓ ПРАВИЛЬНЕ")
            else:
                print("\nПравильність розфарбування: ✗ НЕПРАВИЛЬНЕ")

            # Інтерфейс: verbose режим - детальна інформація про вершини та ребра
            if args.verbose:
                print("\nВершини та їх кольори:")
                color_names = {'r': 'червоний', 'g': 'зелений', 'b': 'синій',
                              'R': 'червоний', 'G': 'зелений', 'B': 'синій'}
                for node in sorted(graph.nodes()):
                    color = graph.nodes[node].get('color', 'не вказано')
                    color_display = color_names.get(color, color)
                    neighbors = list(graph.neighbors(node))
                    print(f"  {node}: колір '{color}' ({color_display}), "
                          f"степінь {graph.degree(node)}, сусідні: {neighbors}")

                print("\nРебра:")
                for u, v in sorted(graph.edges()):
                    print(f"  ({u}, {v})")

            print("="*60 + "\n")

        elif args.mode == "perf":
            # Інтерфейс: режим порівняння ефективності вбудованого та networkx алгоритму.
            print("\n" + "="*60)
            print("ЕФЕКТИВНІСТЬ АЛГОРИТМІВ")
            perfomance(graph)
            print("="*60 + "\n")

    # Інтерфейс: обробка помилок при роботі з файлами та даними
    except FileNotFoundError:
        print(f"\n✗ ПОМИЛКА: Файл '{args.filepath}' не знайдено!")
        print("Перевірте правильність шляху до файлу.")
    except json.JSONDecodeError as e:
        print("\n✗ ПОМИЛКА: Не вдалося розпарсити файл у JSON!")
        print(f"Деталі: {e}")
    except KeyError as e:
        print("\n✗ ПОМИЛКА: Відсутнє обов'язкове поле в JSON файлі!")
        print(f"Деталі: {e}")

if __name__ == "__main__":
    main()
