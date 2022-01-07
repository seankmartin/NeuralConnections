"""Combine atlas operations with graph operations."""

import os
import time
from pprint import pprint
import cProfile, pstats, io
from pstats import SortKey
from skm_pyutils.py_profile import profileit

import numpy as np
from .atlas import gen_graph_for_regions
from .simple_graph import reverse, find_connected_limited
from .nx_graph import nx_vis_force, nx_create_graph
from .connectivity_patterns import MatrixConnectivity
from .matrix import (
    convert_mouse_data,
    load_matrix_data,
    print_args_dict,
    graph_connectome,
    mpf_connectome,
    gen_random_matrix,
)
from .mpf_connection import CombProb

here = os.path.dirname(os.path.abspath(__file__))


def gen_random_matrix_(region_sizes, result):
    ab, ba, aa, bb = gen_random_matrix(*region_sizes, 0.1, 0.2, 0.001, 0.0005)
    mc = MatrixConnectivity(ab=ab, ba=ba, aa=aa, bb=bb)
    mc.create_connections()
    reverse_graph = reverse(mc.graph)
    args_dict = mc.compute_stats()
    result["full_matrix_stats"] = print_args_dict(args_dict, out=False)
    to_write = [mc.num_a, mc.num_b]

    return mc, reverse_graph, to_write, args_dict


def process_matrix_data(A_name, B_name, region_sizes, result):
    """
    Grab the matrix data as a graph for given regions.

    Parameters
    ----------
    A_name : str
        The name of region 1.
    B_name : str
        The name of region 2.
    region_sizes : list of int
        The size of the regions.
    result : dict
        A dictionary to save the results to.

    Returns
    -------
    mc, reverse_graph, to_write, args_dict
        mc : MatrixConnectivity
        reverse_graph : list[list]
        to_write : list of int
        args_dict : dict

    """
    convert_mouse_data(A_name, B_name)
    to_use = [True, True, True, True]
    mc, args_dict = load_matrix_data(to_use, A_name, B_name)
    print("{} - {}, {} - {}".format(A_name, B_name, mc.num_a, mc.num_b))

    if region_sizes is not None:
        print(f"Subsampled regions to {region_sizes}")
        mc = mc.subsample(*region_sizes)
    mc.create_connections()
    args_dict = mc.compute_stats()
    result["full_matrix_stats"] = print_args_dict(args_dict, out=False)
    to_write = [mc.num_a, mc.num_b]
    reverse_graph = reverse(mc.graph)

    return mc, reverse_graph, to_write, args_dict


def compare_sub_and_full(
    mc, reverse_graph, a_indices, b_indices, num_sampled, max_depth=1, num_iters=1
):
    print("Comparing the subsample to the full method.")
    sub_mc, args_dict = mc.compute_probe_stats(a_indices, b_indices)
    sub_mc.create_connections()
    reverse_sub = reverse(sub_mc.graph)

    flat_indices_a = np.arange(len(a_indices))
    flat_indices_b = np.arange(len(b_indices))

    def random_var_gen(iter_val):
        start_idx = np.random.choice(flat_indices_a, size=num_sampled[0], replace=False)
        end_idx = np.random.choice(flat_indices_b, size=num_sampled[1], replace=False)

        start = np.array(a_indices)[start_idx]
        end = np.array(b_indices)[end_idx]
        end = end + mc.num_a

        return start_idx, end_idx, start, end

    def fn_to_eval(start, end):
        return (find_connected_limited(mc.graph, start, end, max_depth, reverse_graph),)

    def sub_fn_to_eval(start, end):
        return (
            find_connected_limited(sub_mc.graph, start, end, max_depth, reverse_sub),
        )

    full_results = []
    for i in range(num_iters):
        start_idx, end_idx, start, end = random_var_gen(i)
        big_res = fn_to_eval(start, end)
        small_res = sub_fn_to_eval(start_idx, end_idx)

        full_results.append(
            dict(
                big=big_res,
                small=small_res,
                start_idx=start_idx,
                end_idx=end_idx,
                start=start,
                end=end,
            )
        )

    return full_results


def atlas_control(
    A_name,
    B_name,
    num_sampled,
    max_depth=1,
    num_iters=100,
    num_cpus=1,
    region_sizes=None,
    atlas_name="allen_mouse_25um",
    session_id=None,
    hemisphere="left",
    load=True,
):
    """
    Full run of atlas merged with graph.

    Parameters
    ----------
    A_name : str
        The name of the first region.
    B_name : str
        The name of the second region.
    num_sampled : list of int
        The number of samples obtained from each region in turn.
    max_depth : int, optional
        The max depth of synapses, by default 1
    num_iters : int, optional
        The number of Monte Carlo simulation, by default 100
    num_cpus : int, optional
        The number of CPUs to use, by default 1
    region_sizes : list of int, optional
        The size of the regions to use in turn, by default None
    atlas_name : str, optional
        The name of the atlas to use, by default "allen_mouse_25um"
    session_id : int, optional
        The ID of the session to use, by default None
    hemisphere : str, optional
        The side of the brain, by default "left"
    load : bool, optional
        Whether to load real data or generate fake data (faster), by default True

    Returns
    -------
    dict
        The set of results

    Raises
    ------
    ValueError
        Not enough points in region A or B to get number of samples

    """
    print("Starting program")
    result = {}

    # 1. Load or generate the connection matrices
    t = time.perf_counter()
    if load:
        mc, reverse_graph, to_write, args_dict = process_matrix_data(
            A_name, B_name, region_sizes, result
        )
        graph = mc.graph
        if region_sizes is None:
            region_sizes = to_write
    else:
        mc, reverse_graph, to_write, args_dict = gen_random_matrix_(
            region_sizes, result
        )
    end_time = time.perf_counter() - t
    print(f"Finished generating or loading matrix data in {end_time:.2f}s")

    # 2. Find the points which lie in the probes
    t = time.perf_counter()
    region_pts, brain_region_meshes, probes_to_use = gen_graph_for_regions(
        [A_name, B_name],
        region_sizes,
        atlas_name=atlas_name,
        session_id=session_id,
        hemisphere=hemisphere,
        sort_=True,
    )
    a_indices = region_pts[0][1]
    b_indices = region_pts[1][1]

    result["Num intersected"] = [len(r[0]) for r in region_pts]
    if result["Num intersected"][0] < num_sampled[0]:
        n = result["Num intersected"][0]
        raise ValueError(
            f"Not enough points found in probe radius for {A_name}" + f"found {n}"
        )
    elif result["Num intersected"][1] < num_sampled[1]:
        n = result["Num intersected"][1]
        raise ValueError(
            f"Not enough points found in probe radius for {B_name}" + f"found {n}"
        )
    end_time = time.perf_counter() - t
    print(f"Finished finding intersections in {end_time:.2f}s")

    # 2A TEMP compare the subsampled versus the full graph
    compare_sub_and_full(
        mc, reverse_graph, a_indices, b_indices, num_sampled, max_depth, num_iters=10
    )
    return

    # 3. Monte carlo simulation on these points
    graph_res = graph_connectome(
        num_sampled,
        max_depth,
        num_iters,
        graph,
        reverse_graph,
        to_write,
        num_cpus,
        a_indices,
        b_indices,
    )
    result["graph"] = graph_res

    # 4. Mathematical calculation on these points
    mc, args_dict = mc.compute_probe_stats(
        a_indices,
        b_indices,
    )
    mpf_res = mpf_connectome(mc, num_sampled, max_depth, args_dict)
    result["mpf"] = mpf_res
    result["probe_matrix_stats"] = print_args_dict(args_dict, out=False)

    # 5a TEMP
    mc.create_connections()
    reverse_graph = reverse(mc.graph)
    graph_res = graph_connectome(
        num_sampled,
        max_depth,
        num_iters,
        mc.graph,
        reverse_graph,
        [mc.num_a, mc.num_b],
        num_cpus,
    )
    result["small_graph"] = graph_res

    # 5b. TEMP? Visualise the small graph inside probes
    nx_graph = nx_create_graph(mc.graph)
    random_sources, random_targets = mc.gen_random_samples(num_sampled, zeroed=False)
    reachable = find_connected_limited(
        mc.graph,
        random_sources,
        random_targets,
        max_depth=max_depth,
        reverse_graph=reverse_graph,
    )
    nx_vis_force(
        nx_graph,
        start_set=mc.a_indices,
        end_set=mc.num_a + mc.b_indices,
        sources=random_sources,
        targets=random_targets,
        name=os.path.join(here, "..", "figures", "nx_atlas.png"),
        labels=False,
        reachable=reachable,
    )

    if result is not None:
        with open(os.path.join(here, "..", "results", "atlas.txt"), "w") as f:
            pprint(result, width=120, stream=f)

    return result


if __name__ == "__main__":
    np.random.seed(42)
    A_name = "VISp"
    B_name = "VISl"
    region_sizes = [10000, 5000]
    atlas_name = "allen_mouse_25um"
    session_id = None
    hemisphere = "left"
    profile = True
    num_sampled = [10, 7]
    max_depth = 2
    num_iters = 100
    load = False

    if profile:

        @profileit("atlas_output.log")
        def run_atlas():
            atlas_control(
                A_name=A_name,
                B_name=B_name,
                num_sampled=num_sampled,
                max_depth=max_depth,
                num_iters=num_iters,
                num_cpus=1,
                region_sizes=region_sizes,
                atlas_name=atlas_name,
                session_id=session_id,
                hemisphere=hemisphere,
                load=load,
            )

        run_atlas()
    else:
        atlas_control(
            A_name=A_name,
            B_name=B_name,
            num_sampled=num_sampled,
            max_depth=max_depth,
            num_iters=num_iters,
            num_cpus=1,
            region_sizes=region_sizes,
            atlas_name=atlas_name,
            session_id=session_id,
            hemisphere=hemisphere,
            load=load,
        )