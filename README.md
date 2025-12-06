This project provides tools for working with undirected graphs and applying a backtracking algorithm to find a valid 3-coloring. It includes:

* loading graphs from JSON,
* validating a coloring,
* visualizing the graph,
* animating the coloring process,
* saving the final result.

All vertices in the input graph must have an initial color.

## Features

* **JSON graph loader**
  Reads nodes, colors, and adjacency lists.

* **Backtracking 3-coloring algorithm**
  Attempts to assign colors `"r"`, `"g"`, `"b"` so that adjacent nodes differ.

* **Visualization**
  Draws the graph using `matplotlib` and colors from each frame.

* **Step-by-step animation**
  Shows how the algorithm explores states and makes decisions.
  Finalized nodes are outlined with a thicker border.

* **Command-line interface**
  Allows choosing between modes such as checking, drawing, coloring, and exporting.


## JSON Format

The input file must follow this structure:

```json
{
  "1": { "color": "r", "edge_with": [2, 3] },
  "2": { "color": "g", "edge_with": [1] },
  "3": { "color": "b", "edge_with": [1] }
}
```

### Rules
* `"color"` is required for every node.
* `"edge_with"` lists all neighbors.
* The graph must be undirected

## Usage

Run the script:

```bash
python FULL_CODE.py input.json --mode <mode> [--output file.json] [--verbose]
```

### Modes

| Mode    | Description                                                    |
| ------- | -------------------------------------------------------------- |
| `info`  | Show basic graph information and validity of initial coloring. |
| `check` | Validate that no adjacent nodes share the same color.          |
| `dict`  | Print a sorted adjacency dictionary.                           |
| `draw`  | Visualize the graph.                                           |
| `color` | Run the backtracking algorithm and animate the process.        |

Example:

```bash
python FULL_CODE.py graph.json --mode color --output result.json
```

---

## Output

If an output file is specified in `--output`, the final coloring is saved in the same JSON format, with `"color"` fields updated based on the solution.

Example:

```json
{
  "1": { "color": "g", "edge_with": [2, 3] },
  "2": { "color": "r", "edge_with": [1] },
  "3": { "color": "b", "edge_with": [1] }
}
```

## Animation

The coloring process is animated using `matplotlib.animation.FuncAnimation`.

* Each frame shows the state of the coloring at a particular step.
* Finalized vertices are drawn with a thicker outline.
* Layout is generated with `spring_layout`.
