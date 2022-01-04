import os
from pprint import pprint
import cProfile, pstats, io
from pstats import SortKey
from skm_pyutils.py_profile import profileit

from .atlas import gen_graph_for_regions
from .simple_graph import reverse
from .matrix import (
    convert_mouse_data,
    load_matrix_data,
    print_args_dict,
    graph_connectome,
)

here = os.path.dirname(os.path.abspath(__file__))


def process_matrix_data(A_name, B_name, region_sizes, result):
    convert_mouse_data(A_name, B_name)
    to_use = [True, True, True, True]
    mc, args_dict = load_matrix_data(to_use, A_name, B_name)
    print("{} - {}, {} - {}".format(A_name, B_name, mc.num_a, mc.num_b))


    if region_sizes is not None:
        print(f"Subsampled regions to {region_sizes}")
        mc = mc.subsample(*region_sizes)
    mc.create_connections()
    args_dict = mc.compute_stats()
    result["matrix_stats"] = print_args_dict(args_dict, out=False)
    graph = mc.graph
    to_write = [mc.num_a, mc.num_b]
    reverse_graph = reverse(graph)

    return graph, reverse_graph, to_write


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
):
    result = {}
    # 1. Load the connection matrices
    graph, reverse_graph, to_write = process_matrix_data(
        A_name, B_name, region_sizes, result
    )
    if region_sizes is None:
        region_sizes = to_write

    # 2. Find the points which lie in the probes
    region_pts, brain_region_meshes, probes_to_use = gen_graph_for_regions(
        [A_name, B_name],
        region_sizes,
        atlas_name=atlas_name,
        session_id=session_id,
        hemisphere=hemisphere,
        sort_=True,
    )
    result["Num intersected"] = [len(r[0]) for r in region_pts]
    if (result["Num intersected"][0] < num_sampled[0]):
        n = result["Num intersected"][0]
        raise ValueError(
            f"Not enough points found in probe radius for {A_name}" + 
            f"found {n}"
            )
    elif (result["Num intersected"][1] < num_sampled[1]):
        n = result["Num intersected"][1]
        raise ValueError(
            f"Not enough points found in probe radius for {B_name}" +
            f"found {n}")

    graph_res = graph_connectome(
        num_sampled,
        max_depth,
        num_iters,
        graph,
        reverse_graph,
        to_write,
        num_cpus,
        region_pts[0][1],
        region_pts[1][1],
    )
    result["graph"] = graph_res

    if result is not None:
        with open(os.path.join(here, "..", "results", "atlas.txt"), "w") as f:
            pprint(result, width=120, stream=f)

    return result


if __name__ == "__main__":
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
            )

        run_atlas()

        # pr = cProfile.Profile()
        # pr.enable()
        # atlas_control(A_name, B_name, region_sizes, atlas_name, session_id, hemisphere)
        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
    else:
        atlas_control(A_name, B_name, region_sizes, atlas_name, session_id, hemisphere)
