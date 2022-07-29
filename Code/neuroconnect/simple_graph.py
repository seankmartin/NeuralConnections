"""
A full description of simple graphs.

Graphs are designed as a list of edges.
Like so:
[1, 2]
[0]
[2]
Would be a graph with three vertices and 
four edges 0->1, 0->2, 1->0 and 2->2.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import find, lil_matrix
from time import perf_counter

from .plot_graph import get_positions, get_colours
from .connect_math import multi_2d_avg, window_2d_avg
from .atlas import get_n_random_points_in_region


def reverse(graph):
    """Reverse the direction of edges in a graph."""
    reverse_graph = [[] for _ in range(len(graph))]

    for node, children in enumerate(graph):
        for child in children:
            reverse_graph[child].append(node)

    return reverse_graph


def from_matrix(AB, BA, AA, BB, to_use=(True, True, True, True)):
    """Return a graph from matrix representation."""
    num_a, num_b = AB.shape
    graph = [[] for _ in range(num_a + num_b)]
    finals = [[], []]

    t1 = perf_counter()
    if to_use[0]:
        nonzeros_row, nonzeros_col = AB.nonzero()
        nonzeros_col = nonzeros_col + num_a
        finals[0].append(nonzeros_row)
        finals[1].append(nonzeros_col)
    if to_use[1]:
        nonzeros_row, nonzeros_col = BA.nonzero()
        nonzeros_row = nonzeros_row + num_a
        finals[0].append(nonzeros_row)
        finals[1].append(nonzeros_col)
    if to_use[2]:
        nonzeros_row, nonzeros_col = AA.nonzero()
        finals[0].append(nonzeros_row)
        finals[1].append(nonzeros_col)
    if to_use[3]:
        nonzeros_row, nonzeros_col = BB.nonzero()
        nonzeros_row = nonzeros_row + num_a
        nonzeros_col = nonzeros_col + num_a
        finals[0].append(nonzeros_row)
        finals[1].append(nonzeros_col)
    final_rows = np.concatenate(finals[0])
    final_cols = np.concatenate(finals[1])
    for x, y in zip(final_rows, final_cols):
        graph[x].append(y)

    t2 = perf_counter()
    tt = t2 - t1
    print(f"Converted matrix in {tt:.2f} seconds")

    return graph


def to_matrix(graph, num_a, num_b):
    """Return a matrix from graph representation."""
    AB = lil_matrix((num_a, num_b))
    BA = lil_matrix((num_b, num_a))
    AA = lil_matrix((num_a, num_a))
    BB = lil_matrix((num_b, num_b))

    for i, target_list in enumerate(graph):
        if i < num_a:
            for j in target_list:
                if j < num_a:
                    AA[i, j] = 1
                else:
                    AB[i, j - num_a] = 1
        else:
            for j in target_list:
                if j < num_a:
                    BA[i - num_a, j] = 1
                else:
                    BB[i - num_a, j - num_a] = 1

    return AB, BA, AA, BB


def find_path(graph, start, end, path=None):
    """Return a path from start to end if it exists."""
    if path is None:
        path = []
    path = path + [start]
    if start == end:
        return path
    if len(graph) < start:
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath is not None:
                return newpath
    return None


def create_graph(region_sizes, connection_strategy, connectivity_params, **kwargs):
    """Create a graph using a connectivity pattern from connectivity_patterns."""
    graph = []
    region_verts = []
    count = 0
    for r_s in region_sizes:
        vertices = [j + count for j in range(r_s)]
        region_verts.append(vertices)
        count += r_s
    for i in range(len(region_sizes)):

        # Recursive connections
        if i == len(region_sizes) - 1:
            connect_inst = connection_strategy(**connectivity_params[i], recursive=True)
            connections, _ = connect_inst.create_connections(
                region_verts[0], region_verts=region_verts[i], **kwargs
            )

        # Regular connections
        else:
            connect_inst = connection_strategy(
                **connectivity_params[i], recursive=False
            )
            connections, connected = connect_inst.create_connections(
                region_verts[i + 1], region_verts=region_verts[i], **kwargs
            )
        graph = graph + connections

    return graph, connected


def create_3d_graph(region_sizes, regions, connection_strategy, connectivity_params):
    """
    Create a 3d graph using a connectivity pattern from connectivity_patterns.

    regions should be a list of vedo meshes or brainrender actors.

    Returns
    -------
    graph
        A graph as a list of lists indicating connections
    connected
        A list of vertices which send connections
    positions
        A list of 3D positions of cells within the region.

    """

    graph = []
    region_verts = []
    count = 0
    for r_s in region_sizes:
        vertices = [j + count for j in range(r_s)]
        region_verts.append(vertices)
        count += r_s
    for i in range(len(region_sizes)):

        positions = get_n_random_points_in_region(regions[i], region_sizes[i], s=None)

        # Recursive connections
        if i == len(region_sizes) - 1:
            connect_inst = connection_strategy(**connectivity_params[i], recursive=True)
            connections, _ = connect_inst.create_connections(
                region_verts[0], region_verts=region_verts[i]
            )

        # Regular connections
        else:
            connect_inst = connection_strategy(
                **connectivity_params[i], recursive=False
            )
            connections, connected = connect_inst.create_connections(
                region_verts[i + 1], region_verts=region_verts[i]
            )
        graph = graph + connections

    return graph, connected, positions


def is_reachable(graph, end, start_set):
    """Return True if end can be reached from any neuron in start_set."""
    for start in start_set:
        result = find_path(graph, start, end)
        if result is not None:
            return True
    return False


def find_connected_limited(graph, start_set, end_set, max_depth=3, reverse_graph=None):
    """Return the neurons in end_set reachable from start_set with limited depth."""
    if reverse_graph is None:
        reverse_graph = reverse(graph)
    reachable = []
    for end in end_set:
        result = iddfs(reverse_graph, end, start_set, max_depth=max_depth)
        if result is not None:
            reachable.append(end)
    return reachable


def find_connected(graph, start_set, end_set):
    """Return the neurons in end_set reachable from start_set."""
    reachable = []
    for end in end_set:
        if is_reachable(graph, end, start_set):
            reachable.append(end)
    return reachable


def vis_graph(graph, region_sizes, start_set, end_set, reachable=None):
    """Visualise the given graph, networkx force vis is generally preferred."""
    x, y = get_positions(region_sizes, as_dict=False)
    c = get_colours(len(graph), start_set, end_set, reachable)

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=c)
    for i, item in enumerate(graph):
        for j in item:
            x_loc_1 = x[i]
            x_loc_2 = x[j]
            y_loc_1 = y[i]
            y_loc_2 = y[j]
            plot_x = [x_loc_1, x_loc_2]
            plot_y = [y_loc_1, y_loc_2]
            ax.plot(plot_x, plot_y, linestyle="dashed", c="k")
            ax.set_ylim(ax.get_ylim()[::-1])
    return fig


def iddfs(graph, source, goals, max_depth=1000):
    """Iterative deeping depth-first search."""
    for depth in range(max_depth + 1):
        found, remaining = dls(graph, source, goals, depth)
        if found is not None:
            return found
        elif not remaining:
            return None
    return None


def dls(graph, source, goals, depth):
    """Depth limited depth-first search."""
    if depth == 0:
        return (source, True) if source in goals else (None, True)
    elif depth > 0:
        any_remaining = False
        for child in graph[source]:
            found, remaining = dls(graph, child, goals, depth - 1)
            if found is not None:
                return (found, True)
            if remaining:
                any_remaining = True
        return (None, any_remaining)


def matrix_vis(AB, BA, AA, BB, k_size, name="matrix_vis.pdf"):
    """Visualise the connection matrix."""
    total = np.sum(np.array(AB.shape))
    here = os.path.dirname(os.path.realpath(__file__))
    out_name = os.path.join(here, "..", "figures", name)

    print("Averaging over {}^2 matrix in {}x{} windows".format(total, k_size, k_size))

    if BA is not None:
        mat = multi_2d_avg(AB, BA, AA, BB, k_size)
    else:
        mat = window_2d_avg(AB, k_size)
    sampled = np.clip(mat * 100, a_min=0.0, a_max=100.0)

    plt.clf()
    plt.imshow(sampled, interpolation="none", cmap="viridis")
    plt.colorbar()

    plt.savefig(out_name, dpi=400)

    plt.clf()
