"""This module handles experiments related to matrix connections."""

import os
import gc
import numpy as np
from collections import OrderedDict
from pprint import pprint
import pickle

from scipy import sparse
from mpmath import mpf
import pandas as pd

from .connectivity_patterns import MatrixConnectivity
from .simple_graph import find_connected_limited
from .mpf_connection import CombProb
from .monte_carlo import (
    monte_carlo,
    list_to_df,
    summarise_monte_carlo,
    get_distribution,
    dist_difference,
)
from .connect_math import get_dist_mean, get_dist_var
from .nx_graph import nx_create_graph, nx_vis_force
from .simple_graph import matrix_vis, reverse

here = os.path.dirname(os.path.realpath(__file__))
pickle_loc = os.path.abspath(os.path.join(here, "..", "resources", "graph.pickle"))


def convert_mouse_data(A_name, B_name, hemisphere="right"):
    """Convert general blue brain data into smaller data."""
    here = os.path.dirname(os.path.realpath(__file__))
    if hemisphere == "right":
        resource_dir = os.path.join(here, "..", "resources", "right_hemisphere")
    elif hemisphere == "left":
        resource_dir = os.path.join(here, "..", "resources", "left_hemisphere")

    def load_name(name):
        return os.path.join(resource_dir, name)

    def save_name(name):
        return os.path.join(resource_dir, name + ".npz")

    if os.path.isfile(
        os.path.join(resource_dir, "{}_to_{}.npz".format(A_name, B_name))
    ):
        print(f"Already converted this mouse data for {hemisphere} hemisphere")
        return
    print("Pulling out data from the mouse connectome")

    end_bit_indices = "_ALL_INPUTS_ipsi.indices.npy"
    end_bit_out = "_ALL_INPUTS_ipsi.csc.npz"
    end_bit_local = "_ALL_INPUTS_local.csc.npz"

    # Load the relevant data
    A_indices = np.load(load_name(A_name + end_bit_indices))
    B_indices = np.load(load_name(B_name + end_bit_indices))

    A_local = sparse.load_npz(load_name(A_name + end_bit_local))
    # In case some stray indices are left around
    A_small = A_local[A_indices]
    sparse.save_npz(save_name(A_name + "_local"), A_small)

    B_local = sparse.load_npz(load_name(B_name + end_bit_local))
    # In case some stray indices are left around
    B_small = B_local[B_indices]
    sparse.save_npz(save_name(B_name + "_local"), B_small)

    A = sparse.load_npz(load_name(A_name + end_bit_out))
    B_to_A = A[B_indices]
    sparse.save_npz(save_name(B_name + "_to_" + A_name), B_to_A)

    B = sparse.load_npz(load_name(B_name + end_bit_out))
    A_to_B = B[A_indices]
    sparse.save_npz(save_name(A_name + "_to_" + B_name), A_to_B)

    return


def load_matrix_data(to_use, A_name, B_name, hemisphere="right"):
    """
    Load matrix data into a connectivity object.

    Parameters
    ----------
    to_use : list of bool
        Which matrices to consider, in the order [ab, ba, aa, bb]

    """
    here = os.path.dirname(os.path.realpath(__file__))
    if hemisphere == "right":
        resource_dir = os.path.join(here, "..", "resources", "right_hemisphere")
    elif hemisphere == "left":
        resource_dir = os.path.join(here, "..", "resources", "left_hemisphere")

    mc = MatrixConnectivity(
        ab=os.path.join(resource_dir, "{}_to_{}.npz".format(A_name, B_name)),
        ba=os.path.join(resource_dir, "{}_to_{}.npz".format(B_name, A_name)),
        aa=os.path.join(resource_dir, "{}_local.npz".format(A_name)),
        bb=os.path.join(resource_dir, "{}_local.npz".format(B_name)),
        to_use=to_use,
    )
    args_dict = mc.compute_stats()

    return mc, args_dict


def mpf_connectome(
    mc, num_sampled, max_depth, args_dict, clt_start=10, sr=0.01, mean_estimate=False
):
    """Perform mpf statistical calculations on the mouse connectome."""
    args_dict["max_depth"] = max_depth
    args_dict["total_samples"] = num_sampled[0]
    args_dict["static_verbose"] = False
    args_dict["clt_start"] = clt_start
    args_dict["mean_estimate"] = mean_estimate

    if max_depth > 1:
        sr = None
    if mean_estimate is True:
        sr = None

    cp = CombProb(
        mc.num_a,
        num_sampled[0],
        mc.num_senders,
        mc.num_b,
        num_sampled[1],
        MatrixConnectivity.static_expected_connections,
        verbose=True,
        subsample_rate=sr,
        **args_dict,
    )
    result = {
        "expected": cp.expected_connections(),
        "total": cp.get_all_prob(),
        "each_expected": {k: cp.expected_total(k) for k in range(num_sampled[0] + 1)},
    }

    return result


def mpf_probe_connectome(
    mc,
    num_sampled,
    a_indices,
    b_indices,
    max_depth,
    args_dict,
    clt_start=10,
    sr=0.01,
    mean_estimate=False,
    force_no_mean=False,
):
    """Perform mpf statistical calculations on the mouse connectome with a probe."""
    probe_stats = mc.compute_probe_stats(
        a_indices,
        b_indices,
    )
    sub_mc = probe_stats["probes"]
    sub_args_dict = probe_stats["stats"]
    for k, v in sub_args_dict.items():
        args_dict[f"{k}_probe"] = v
    sub_args_dict = probe_stats["A_stats"]
    for k, v in sub_args_dict.items():
        args_dict[f"{k}_A"] = v
    sub_args_dict = probe_stats["B_stats"]
    for k, v in sub_args_dict.items():
        args_dict[f"{k}_B"] = v
    sub_args_dict = probe_stats["inter"]
    for k, v in sub_args_dict.items():
        args_dict[k] = v

    args_dict["max_depth"] = max_depth
    args_dict["total_samples"] = num_sampled[0]
    args_dict["static_verbose"] = False
    args_dict["clt_start"] = clt_start
    args_dict["mean_estimate"] = mean_estimate

    if force_no_mean:
        args_dict["use_mean"] = False

    if max_depth > 1:
        sr = None
    if mean_estimate is True:
        sr = None

    # This comb prob is for depth =-1
    # If you call senders dist it only works for dpeth = 1
    # Inside of the detla function static_expected_connections
    # Ti instead works correctly for dpth != 1
    cp = CombProb(
        sub_mc.num_a,
        num_sampled[0],
        sub_mc.num_senders,
        sub_mc.num_b,
        num_sampled[1],
        MatrixConnectivity.static_expected_connections,
        verbose=True,
        subsample_rate=sr,
        **args_dict,
    )
    result = {
        "expected": cp.expected_connections(),
        "total": cp.get_all_prob(),
        "each_expected": {k: cp.expected_total(k) for k in range(num_sampled[0] + 1)},
    }

    return result


def handle_pickle(data, name, mode):
    """Save or load data to a pickle file."""
    pickle_name = os.path.join(os.path.dirname(pickle_loc), name)
    if mode == "w":
        with open(pickle_name, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    if mode == "r":
        with open(pickle_name, "rb") as handle:
            b = pickle.load(handle)
        return b


def graph_connectome(
    num_sampled,
    max_depth,
    num_iters=10,
    graph=None,
    reverse_graph=None,
    to_write=None,
    num_cpus=1,
    a_indices=None,
    b_indices=None,
):
    """Perform monte carlo graph simulations on the mouse connectome."""
    num_a, num_b = to_write

    if a_indices is None:
        a_indices = np.array([i for i in range(num_a)])
    if b_indices is None:
        b_indices = np.array([i for i in range(num_b)])

    def random_var_gen(iter_val):
        start = np.random.choice(a_indices, size=num_sampled[0], replace=False)
        end = np.random.choice(b_indices, size=num_sampled[1], replace=False)
        end = end + num_a

        return start, end

    def fn_to_eval(start, end):
        return (
            len(find_connected_limited(graph, start, end, max_depth, reverse_graph)),
        )

    result = monte_carlo(fn_to_eval, random_var_gen, num_iters, num_cpus=num_cpus)
    df = list_to_df(
        result,
        ["Connections"],
    )
    result = summarise_monte_carlo(
        df,
        plot=False,
    )
    ordered_dist = get_distribution(df, "Connections", num_iters)

    return {
        "full_results": df,
        "summary_stats": result,
        "dist": ordered_dist,
    }


def print_args_dict(args_dict, out=True):
    to_print = {}
    for v in ("N", "num_start", "num_senders", "num_recurrent"):
        to_print[v] = args_dict[v]
    for v in (
        "out_connections_dist",
        "recurrent_connections_dist",
        "start_inter_dist",
        "end_inter_dist",
    ):
        to_print[v] = (get_dist_mean(args_dict[v]), get_dist_var(args_dict[v]))

    if out:
        print(to_print)

    return to_print


def check_stats(mc, div_ratio, max_depth, num_iters=1000, num_cpus=1, plot=False):
    """Compare the results from simulations and stats."""
    a_samples, b_samples = (int(mc.num_a / div_ratio), int(mc.num_b / div_ratio))
    div = (30, 10)  # num_samples / x per region
    new_mc = mc.subsample(a_samples, b_samples)
    new_mc.create_connections()
    args_dict = new_mc.compute_stats()
    print_args_dict(args_dict, out=True)

    num_samples = np.ceil([a_samples / div[0], b_samples / div[1]]).astype(np.int32)
    print("Sampling {}".format(num_samples))

    if plot:
        nx_graph = nx_create_graph(new_mc.graph)
        start, end = new_mc.gen_random_samples(num_samples, zeroed=False)
        os.makedirs(os.path.join(here, "..", "figures"), exist_ok=True)
        nx_vis_force(
            nx_graph,
            new_mc.a_indices,
            new_mc.num_a + new_mc.b_indices,
            start,
            end,
            name=os.path.join(here, "..", "figures", "mouse_graph_small.png"),
        )

    def random_var_gen(iter_val):
        start, end = new_mc.gen_random_samples(num_samples, zeroed=False)
        return start, end

    def fn_to_eval(start, end):
        reachable = find_connected_limited(
            new_mc.graph, start, end, max_depth=max_depth
        )
        return (len(reachable),)

    # Stats check
    args_dict["max_depth"] = max_depth
    args_dict["total_samples"] = num_samples[0]
    args_dict["static_verbose"] = False
    cp = CombProb(
        new_mc.num_a,
        float(num_samples[0]),
        new_mc.num_senders,
        new_mc.num_b,
        float(num_samples[1]),
        MatrixConnectivity.static_expected_connections,
        verbose=False,
        **args_dict,
    )
    result_mpf = {
        "expected": cp.expected_connections(),
        "total": cp.get_all_prob(),
    }

    result = monte_carlo(fn_to_eval, random_var_gen, num_iters, num_cpus=num_cpus)
    df = list_to_df(
        result,
        ["Connections"],
    )
    result = summarise_monte_carlo(
        df,
        plot=False,
    )
    ordered_dist = get_distribution(df, "Connections", num_iters)

    return {
        "full_results": df,
        "summary_stats": result,
        "dist": ordered_dist,
        "mpf": result_mpf,
        "difference": dist_difference(result_mpf["total"], ordered_dist),
    }


def make_full_matrix(AB, BA, AA, BB):
    """Combine separate into one sparse matrix."""
    a_size, b_size = AB.shape

    full_mat = sparse.csc_matrix((a_size + b_size, a_size + b_size))
    full_mat[:a_size, :a_size] = AA
    full_mat[:a_size, a_size:] = AB
    full_mat[a_size:, :a_size] = BA
    full_mat[a_size:, a_size:] = BB

    return full_mat


def gen_random_matrix(a_size, b_size, AB_d, BA_d, AA_d, BB_d):
    """Generate a full connectivity matrix at random."""

    def random_gen(n_samples):
        return np.ones(shape=(n_samples,))

    AB = sparse.random(a_size, b_size, AB_d, data_rvs=random_gen, format="csr")
    BA = sparse.random(b_size, a_size, BA_d, data_rvs=random_gen, format="csr")
    AA = sparse.random(a_size, a_size, AA_d, data_rvs=random_gen, format="csr")
    BB = sparse.random(b_size, b_size, BB_d, data_rvs=random_gen, format="csr")

    return AB, BA, AA, BB


def main(
    num_sampled=[3, 3],
    max_depth=2,
    num_iters=1000,
    do_graph=False,
    # These are for checking stats on smaller data
    subsample=False,
    plot=False,
    # Generates a random matrix for comparison
    random=False,
    # Visualise the connection matrix
    vis_connect=False,
    subsample_vis=False,
    # Generate final graphs
    final=False,
    # Analyse
    analyse=False,
    only_exp=False,
    # Which regions are considered here
    # A_name, B_name = "MOp", "SSP-ll"
    A_name="VISp",
    B_name="VISl",
    desired_depth=1,
    desired_samples=79,
):
    """Load data and perform calculations."""
    np.random.seed(42)

    if random:
        AB, BA, AA, BB = gen_random_matrix(150, 50, 0, 0.04, 0, 0.0)
        matrix_vis(AB, BA, AA, BB, 10, name="test_vis.png")

    os.makedirs(os.path.dirname(pickle_loc), exist_ok=True)
    convert_mouse_data(A_name, B_name)
    to_use = [True, True, True, True]
    mc, args_dict = load_matrix_data(to_use, A_name, B_name)
    print("{} - {}, {} - {}".format(A_name, B_name, mc.num_a, mc.num_b))

    result = {}
    result["matrix_stats"] = print_args_dict(args_dict, out=False)

    if only_exp:
        mpf_res = mpf_connectome(mc, num_sampled, max_depth, args_dict)
        mpf_val = [
            mpf_res["expected"],
            mpf_res["expected"] / num_sampled[1],
            "{}_{}".format(A_name, B_name),
            "Statistical estimation",
        ]
        if do_graph:
            print("Converting matrix")
            gc.collect()
            mc.create_connections()
            print("Finished conversion")
            graph = mc.graph
            to_write = [mc.num_a, mc.num_b]
            del mc
            gc.collect()
            reverse_graph = reverse(graph)
            graph_res = graph_connectome(
                num_sampled,
                max_depth,
                graph=graph,
                reverse_graph=reverse_graph,
                to_write=to_write,
                num_iters=num_iters,
            )
            to_add = np.mean(graph_res["full_results"]["Connections"].values)
            graph_val = [
                to_add,
                to_add / num_sampled[1],
                "{}_{}".format(A_name, B_name),
                "Statistical estimation",
            ]
            return mpf_val, graph_val
        return mpf_val, None

    # Convert to a pickle
    # if not os.path.isfile(pickle_loc):
    #     print("Converting matrix")
    #     gc.collect()
    #     mc.create_connections()
    #     print("Finished conversion")
    #     graph = mc.graph
    #     to_write = [mc.num_a, mc.num_b]
    #     del mc
    #     gc.collect()

    #     handle_pickle(graph, "graph.pickle", "w")
    #     handle_pickle(reverse(graph), "r_graph.pickle", "w")
    #     handle_pickle(to_write, "graph_size.pickle", "w")

    if vis_connect:
        if subsample_vis:
            print("Plotting subsampled matrix vis")
            new_mc = mc.subsample(int(mc.num_a / 10), int(mc.num_b / 10))
            matrix_vis(
                new_mc.ab,
                new_mc.ba,
                new_mc.aa,
                new_mc.bb,
                15,
                name="mc_mat_vis_sub10.pdf",
            )
        else:
            o_name = "mc_mat_vis_{}_to_{}.pdf".format(A_name, B_name)
            print("Plotting full matrix vis")
            matrix_vis(mc.ab, mc.ba, mc.aa, mc.bb, 150, name=o_name)
        print("done vis")

    print(mc, print_args_dict(args_dict, out=False))

    result = None
    if subsample:
        result = check_stats(mc, 1000, 1, 20000, 1, plot)
    if final:
        result = {}

        # For different depths and number of samples
        for depth in range(1, 4):
            for ns in range(1, num_sampled[0] + 1):
                ns_2 = [ns] * 2
                mpf_res = mpf_connectome(mc, ns_2, depth, args_dict)
                result["mpf_{}_{}".format(depth, ns)] = mpf_res

        # Save this for plotting
        cols = ["Number of samples", "Proportion of connections", "Max distance"]
        depth_name = [None, "Direct synapse", "Two synapses", "Three synapses"]
        vals = []
        for depth in range(1, 4):
            for ns in range(1, num_sampled[0] + 1):
                this = result["mpf_{}_{}".format(depth, ns)]
                val = [ns, this["expected"] / ns, depth_name[depth]]
                vals.append(val)
        df = pd.DataFrame(vals, columns=cols)
        os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
        df.to_csv(
            os.path.join(
                here, "..", "results", "{}_to_{}_depth.csv".format(A_name, B_name)
            ),
            index=False,
        )

        cols = ["Number of sampled connected neurons", "Probability"]
        total_pmf = result["mpf_{}_{}".format(desired_depth, desired_samples)]["total"]
        vals = []
        for k, v in total_pmf.items():
            vals.append([k, float(v)])
        df = pd.DataFrame(vals, columns=cols)
        df.to_csv(
            os.path.join(
                here,
                "..",
                "results",
                "{}_to_{}_pmf_{}_{}.csv".format(
                    A_name, B_name, desired_depth, desired_samples
                ),
            ),
            index=False,
        )
    if analyse:
        result = {}
        result["matrix_stats"] = args_dict

        mpf_res = mpf_connectome(
            mc,
            num_sampled,
            max_depth,
            args_dict,
            clt_start=30,
            sr=None,
            mean_estimate=True,
        )
        result["mean"] = mpf_res

        vals = []
        cols = ["Number of connected neurons", "Probability", "Calculation"]
        for k, v in mpf_res["total"].items():
            vals.append([k, float(v), "Mean estimation"])

        mpf_res = mpf_connectome(mc, num_sampled, max_depth, args_dict, clt_start=30)
        result["mpf"] = mpf_res

        for k, v in mpf_res["total"].items():
            vals.append([k, float(v), "Statistical estimation"])

        if do_graph:
            print("Converting matrix")
            gc.collect()
            mc.create_connections()
            print("Finished conversion")
            graph = mc.graph
            to_write = [mc.num_a, mc.num_b]
            del mc
            gc.collect()
            reverse_graph = reverse(graph)

            graph_res = graph_connectome(
                num_sampled,
                max_depth,
                graph=graph,
                reverse_graph=reverse_graph,
                to_write=to_write,
                num_iters=num_iters,
            )

            result["difference"] = (
                dist_difference(mpf_res["total"], graph_res["dist"]),
            )
            result["graph"] = graph_res

            for k, v in graph_res["dist"].items():
                vals.append([k, float(v), "Monte Carlo simulation"])

        df = pd.DataFrame(vals, columns=cols)
        df.to_csv(
            os.path.join(
                here,
                "..",
                "results",
                "{}_to_{}_pmf_final_{}_{}.csv".format(
                    A_name, B_name, max_depth, num_sampled[0]
                ),
            ),
            index=False,
        )

    if result is not None:
        with open(os.path.join(here, "..", "results", "mouse.txt"), "w") as f:
            pprint(result, width=120, stream=f)

    return result


if __name__ == "__main__":
    main()
