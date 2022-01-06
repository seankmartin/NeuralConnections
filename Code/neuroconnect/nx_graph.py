"""Defining graphs using the networkx package."""
import os

import networkx
import numpy as np
import matplotlib.pyplot as plt

from .plot_graph import get_positions, get_colours, get_colours_extend


def nx_create_graph(graph):
    """Covert a graph from the simple_graph package into networkx format."""
    G = networkx.DiGraph()
    G.add_nodes_from(range(0, len(graph)))
    for i, edge_list in enumerate(graph):
        if len(edge_list) != 0:
            to_add = [(i, e) for e in edge_list]
            G.add_edges_from(to_add)
    return G


def nx_vis_graph(
    nx_graph, region_sizes, start_set, end_set, reachable=None, name="nx_graph.png"
):
    """Visualise the graph with positions in regions linearly spaced."""
    pos = get_positions(region_sizes, as_dict=True)
    c = get_colours(nx_graph.number_of_nodes(), start_set, end_set, reachable)
    plt.clf()
    networkx.draw_networkx(nx_graph, with_labels=False, node_color=c, pos=pos)
    plt.savefig(name)


def nx_vis_force(
    nx_graph,
    start_set,
    end_set,
    sources,
    targets,
    name="nx_simple.png",
    labels=False,
    reachable=None,
):
    """Simple force based visual representation of the networkx graph."""
    options = {
        "node_size": 50,
        "linewidths": 0,
        "width": 0.1,
    }
    c = get_colours_extend(
        nx_graph.number_of_nodes(), start_set, end_set, sources, targets, reachable
    )
    plt.clf()
    networkx.draw(nx_graph, node_color=c, with_labels=labels, **options)
    plt.savefig(name, dpi=400)


def nx_graph_stats(G):
    """Print simple stats about the graph G."""
    print("Total number of nodes: ", int(G.number_of_nodes()))
    print("Total number of edges: ", int(G.number_of_edges()))
    print("Total number of self-loops: ", int(G.number_of_selfloops()))
    print("List of all nodes with self-loops: ", list(G.nodes_with_selfloops()))
    print(
        "List of all nodes we can go to in a single step from node 2: ",
        list(G.successors(2)),
    )


def nx_find_connected(graph, start_set, end_set, cutoff=np.inf):
    """Return the nodes in end_set connected to start_set."""
    reachable = []
    for end in end_set:
        if nx_is_reachable(graph, end, start_set):
            reachable.append(end)
            if len(reachable) >= cutoff:
                break
    return reachable


def nx_is_reachable(graph, end, start_set):
    """Return if there is path from start_set to end in graph."""
    for start in start_set:
        result = networkx.algorithms.shortest_paths.generic.has_path(graph, start, end)
        if result:
            return True
    return False


def nx_find_connected_limited(graph, start_set, end_set, max_depth=3):
    """Return the neurons in end_set reachable from start_set with limited depth."""
    reverse_graph = graph.reverse()
    reachable = []
    for e in end_set:

        preorder_nodes = list(
            (
                networkx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(
                    reverse_graph, source=e, depth_limit=max_depth
                )
            )
        )

        for s in start_set:
            if s in preorder_nodes:
                reachable.append(e)
                break

    return reachable


def export_gml(graph, name="nx_gml.graphml"):
    """Export the networkx graph to gml format for Gephi visualisation."""
    here = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(here, "..", "resources", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    networkx.write_graphml(graph, path)
    return
