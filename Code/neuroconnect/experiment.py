"""Defining a full experiment of simulating recording neurons."""

import os
from time import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mpf
import tqdm

from .mpf_connection import CombProb
from .simple_graph import (
    create_graph,
    vis_graph,
    find_connected,
    find_connected_limited,
    to_matrix,
    matrix_vis,
    reverse,
)
from .nx_graph import (
    nx_create_graph,
    nx_vis_graph,
    nx_find_connected,
    nx_find_connected_limited,
    nx_vis_force,
    export_gml,
)
from .monte_carlo import (
    monte_carlo,
    list_to_df,
    summarise_monte_carlo,
    get_distribution,
)
from .connect_math import create_uniform


def do_full_experiment(
    region_sizes,
    connection_strategy,
    connectivity_params,
    num_sampled,
    do_mpf=False,
    do_graph=False,
    do_nx=False,
    do_vis_graph=False,
    do_only_none=False,
    num_iters=1000,
    num_cpus=1,
    max_depth=3,
    do_fixed=-1,
    gen_graph_each_iter=False,
    do_mat_vis=False,
    save_every=1,
    quiet=False,
    clt_start=30,
    subsample_rate=0.01,
    approx_hypergeo=False,
    use_mean=True,
):
    """
    Run mpf, simple graphs, and networkx graphs.

    Run a combination of mpf probability, and simple graphs or networkx graphs with
    Monte Carlo simulations.
    """

    results = {}
    overall_start_time = time()
    if do_mpf:
        result = mpf_control(
            region_sizes,
            connection_strategy,
            connectivity_params,
            num_sampled,
            do_only_none,
            max_depth,
            clt_start,
            verbose=(do_fixed != -1),
            subsample_rate=subsample_rate,
            approx_hypergeo=approx_hypergeo,
            use_mean=use_mean,
        )
        results["mpf"] = result

    if do_graph:
        do_nx_vis = do_nx

        if do_fixed != -1:
            for i in range(num_sampled[0] + 1):
                result = graph_control(
                    region_sizes,
                    connection_strategy,
                    connectivity_params,
                    num_sampled,
                    num_iters,
                    do_vis_graph,
                    do_nx_vis,
                    num_cpus=num_cpus,
                    max_depth=max_depth,
                    fixed_samples=i,
                    gen_graph_each_iter=gen_graph_each_iter,
                    do_mat_vis=False,
                    save_every=1,
                    quiet=True,
                )
                results["g_{}".format(i)] = result
        else:
            result = graph_control(
                region_sizes,
                connection_strategy,
                connectivity_params,
                num_sampled,
                num_iters,
                do_vis_graph,
                do_nx_vis,
                num_cpus=num_cpus,
                max_depth=max_depth,
                fixed_samples=-1,
                do_mat_vis=do_mat_vis,
                gen_graph_each_iter=gen_graph_each_iter,
                save_every=save_every,
                quiet=quiet,
            )
            results["graph"] = result

    if do_nx:
        do_nx_vis = (not do_graph) and do_vis_graph
        result = nx_control(
            region_sizes,
            connection_strategy,
            connectivity_params,
            num_sampled,
            num_iters,
            do_nx_vis,
            False,
            num_cpus=num_cpus,
            max_depth=max_depth,
        )
        results["nx"] = result
    elif do_vis_graph:
        nx_control(
            region_sizes,
            connection_strategy,
            connectivity_params,
            num_sampled,
            num_iters,
            False,
            True,
        )

    if (not do_mpf) and (not do_graph) and (not do_nx) and (not do_vis_graph):
        compare_simple_nx(
            region_sizes,
            connection_strategy,
            connectivity_params,
            num_sampled,
            num_iters,
            max_depth=max_depth,
        )

    if not quiet:
        print(
            "Completed everything in {:.2f} seconds".format(time() - overall_start_time)
        )
    return results


def mpf_control(
    region_sizes,
    connection_strategy,
    connectivity_params,
    num_sampled,
    do_only_none,
    max_depth,
    clt_start,
    verbose=False,
    subsample_rate=0.01,
    approx_hypergeo=False,
    use_mean=True,
):
    """Create an mpf CombProb object and perform stats calculations."""
    start_time = time()
    if len(region_sizes) != 2:
        raise ValueError("MPF currently only supports 2 regions")

    if (connection_strategy.__name__ == "RecurrentConnectivity") or (
        connection_strategy.__name__ == "MeanRecurrentConnectivity"
        or (connection_strategy.__name__ == "UniqueConnectivity")
    ):
        unif_out = create_uniform(
            connectivity_params[0]["min_forward"], connectivity_params[0]["max_forward"]
        )
        unif_re = create_uniform(
            connectivity_params[1]["min_forward"], connectivity_params[1]["max_forward"]
        )
        a, b = (
            int(round(region_sizes[0] * connectivity_params[0]["min_inter"])),
            int(round(region_sizes[0] * connectivity_params[0]["max_inter"])),
        )
        inter_a = create_uniform(a, b)
        a, b = (
            int(round(region_sizes[1] * connectivity_params[1]["min_inter"])),
            int(round(region_sizes[1] * connectivity_params[1]["max_inter"])),
        )
        inter_b = create_uniform(a, b)
        delta_params = {
            "out_connections_dist": unif_out,
            "recurrent_connections_dist": unif_re,
            "num_senders": connectivity_params[0]["num_senders"],
            "num_recurrent": connectivity_params[1]["num_senders"],
            "num_start": region_sizes[0],
            "total_samples": num_sampled[0],
            "start_inter_dist": inter_a,
            "end_inter_dist": inter_b,
            "static_verbose": verbose,
            "max_depth": max_depth,
            "init_delta": True,
            "clt_start": clt_start,
            "use_mean": use_mean,
        }

    if connection_strategy.__name__ == "MeanRecurrentConnectivity":
        subsample_rate = None
    if (max_depth > 1) and use_mean:
        subsample_rate = None

    connection_prob = CombProb(
        region_sizes[0],
        num_sampled[0],
        connectivity_params[0]["num_senders"],
        region_sizes[1],
        num_sampled[1],
        connection_strategy.static_expected_connections,
        subsample_rate=subsample_rate,
        approx_hypergeo=approx_hypergeo,
        N=region_sizes[1],
        verbose=verbose,
        **delta_params
    )
    if do_only_none:
        result = {"zero": connection_prob.connection_prob(0)}
    else:
        result = {
            "expected": connection_prob.expected_connections(),
            "total": connection_prob.get_all_prob(),
            "each_expected": {
                k: connection_prob.expected_total(k) for k in range(num_sampled[0] + 1)
            },
        }
    return result


def graph_control(
    region_sizes,
    connection_strategy,
    connectivity_params,
    num_sampled,
    num_iters,
    do_vis_graph,
    do_nx_vis,
    num_cpus=1,
    max_depth=5,
    fixed_samples=-1,
    do_mat_vis=False,
    gen_graph_each_iter=False,
    save_every=1,
    quiet=False,
):
    """
    Create a simple graph and perform monte carlo.

    fixed_samples denotes to pick a fixed number of N
    neurons in region A that send forward connections to B,
    and consider all neurons in B to be target neurons.
    """
    source_vert_list = [i for i in range(region_sizes[0])]
    target_vert_list = [i for i in range(region_sizes[-1])]

    if not gen_graph_each_iter:
        g_graph, g_connected = create_graph(
            region_sizes, connection_strategy, connectivity_params
        )
        g_reverse_graph = reverse(g_graph)
    else:
        g_graph, g_connected, g_reverse_graph = (None, None, None)

    def random_var_gen(iter_val):
        if gen_graph_each_iter:
            graph, connected = create_graph(
                region_sizes, connection_strategy, connectivity_params
            )
        else:
            graph, connected = g_graph, g_connected

        if fixed_samples != -1:
            sources = np.append(
                np.random.choice(connected, fixed_samples, replace=False),
                np.random.choice(
                    np.delete(source_vert_list, connected),
                    num_sampled[0] - fixed_samples,
                    replace=False,
                ),
            )
            targets = np.sum(np.array(region_sizes[:-1])) + np.array(target_vert_list)

        else:
            sources = np.random.choice(source_vert_list, num_sampled[0], replace=False)
            targets = np.sum(np.array(region_sizes[:-1])) + np.random.choice(
                target_vert_list, num_sampled[-1], replace=False
            )
        return graph, sources, targets

    def fn_to_eval(graph, sources, targets):
        if gen_graph_each_iter:
            reverse_graph = None
        else:
            reverse_graph = g_reverse_graph

        reachable = find_connected_limited(
            graph, sources, targets, max_depth=max_depth, reverse_graph=reverse_graph
        )
        # reachable = find_connected(graph, sources, targets)
        return (len(reachable),)

    result = monte_carlo(
        fn_to_eval,
        random_var_gen,
        num_iters,
        num_cpus=num_cpus,
        headers=["Connections",],
        save_name="graph_mc.csv",
        save_every=save_every,
        progress=not quiet,
    )
    df = list_to_df(result, ["Connections",],)
    result = summarise_monte_carlo(
        df, to_plot=["Connections",], plt_outfile="graph_dist.png",
    )
    distrib = get_distribution(df, "Connections", num_iters)

    if do_vis_graph:
        print("Starting graph visualisation")
        graph, sources, targets = random_var_gen(0)
        reachable = find_connected(graph, sources, targets)
        fig = vis_graph(graph, region_sizes, sources, targets, reachable=reachable)
        fig.savefig("graph.png")

        if do_nx_vis:
            graph = nx_create_graph(graph)
            reachable = nx_find_connected(graph, sources, targets)
            nx_vis_graph(graph, region_sizes, sources, targets, reachable=reachable)

    if do_mat_vis:
        print("Starting matrix visualisation")
        graph, _, _ = random_var_gen(0)
        AB, BA, AA, BB = to_matrix(graph, region_sizes[0], region_sizes[-1])
        matrix_vis(AB, BA, AA, BB, k_size=150, name="graph_mat_vis.pdf")

    return {"full_results": df, "summary_stats": result, "dist": distrib}


def nx_control(
    region_sizes,
    connection_strategy,
    connectivity_params,
    num_sampled,
    num_iters,
    do_vis_graph,
    new_vis,
    num_cpus=1,
    max_depth=5,
):
    """Create a networkx graph and perform monte carlo."""
    source_vert_list = [i for i in range(region_sizes[0])]
    target_vert_list = [i for i in range(region_sizes[-1])]

    def random_var_gen(iter_val):
        graph, _ = create_graph(region_sizes, connection_strategy, connectivity_params)
        graph = nx_create_graph(graph)
        sources = np.random.choice(source_vert_list, num_sampled[0], replace=False)
        targets = np.sum(np.array(region_sizes[:-1])) + np.random.choice(
            target_vert_list, num_sampled[-1], replace=False
        )
        return graph, sources, targets

    def fn_to_eval(graph, sources, targets):
        reachable = nx_find_connected_limited(
            graph, sources, targets, max_depth=max_depth
        )
        return (len(reachable),)

    if new_vis:
        graph, sources, targets = random_var_gen(0)
        export_gml(graph)
        here = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(os.path.join(here, "..", "figures"), exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        name = os.path.join(here, "..", "figures", "nx_simple_{}.png".format(now))
        nx_vis_force(
            graph, source_vert_list, target_vert_list, sources, targets, name=name
        )

        return

    result = monte_carlo(fn_to_eval, random_var_gen, num_iters, num_cpus=num_cpus)
    df = list_to_df(result, ["Connections",],)
    result = summarise_monte_carlo(
        df, to_plot=["Connections",], plt_outfile="nx_graph_dist.png",
    )

    if do_vis_graph:
        plt.clf()
        graph, sources, targets = random_var_gen(0)
        reachable = nx_find_connected(graph, sources, targets)
        nx_vis_graph(graph, region_sizes, sources, targets, reachable=reachable)

    return {"full_results": df, "summary_stats": result}


def compare_simple_nx(
    region_sizes,
    connection_strategy,
    connectivity_params,
    num_sampled,
    num_iters,
    max_depth=5,
):
    """
    Compare the results from networkx and the simple graph.

    Note
    ----
    In a small number of cases the results differ, but simple
    graph has been verified to be correct by hand.

    """
    source_vert_list = [i for i in range(region_sizes[0])]
    target_vert_list = [i for i in range(region_sizes[-1])]

    def random_var_gen(iter_val):
        graph, _ = create_graph(region_sizes, connection_strategy, connectivity_params)
        nx_graph = nx_create_graph(graph)
        sources = np.random.choice(source_vert_list, num_sampled[0], replace=False)
        targets = np.sum(np.array(region_sizes[:-1])) + np.random.choice(
            target_vert_list, num_sampled[-1], replace=False
        )
        return graph, nx_graph, sources, targets

    def fn_to_eval(graph, nx_graph, sources, targets):
        reachable = nx_find_connected_limited(
            nx_graph, sources, targets, max_depth=max_depth
        )
        reachable2 = find_connected_limited(
            graph, sources, targets, max_depth=max_depth
        )
        return (reachable, reachable2)

    results = []
    print("Checking if networkx is same as simple graph (netx, simple)")
    good = True
    for i in tqdm.tqdm(range(num_iters)):
        args = random_var_gen(i)
        result = fn_to_eval(*args)
        results.append(result)
        if result[0] != result[1]:
            good = False

            print("Unequal results {}".format(result))
            if region_sizes[0] <= 100:
                print("Plotting situation")
                fig = vis_graph(
                    args[0], region_sizes, args[2], args[3], reachable=result[0]
                )
                fig.savefig("graph_test.png")
                nx_vis_force(
                    args[1],
                    [i for i in range(region_sizes[0])],
                    [
                        np.sum(np.array(region_sizes[:-1])) + i
                        for i in range(region_sizes[1])
                    ],
                    args[2],
                    args[3],
                    name="nx_graph_test.png",
                    labels=True,
                )

                print("Source {}, target {}".format(list(args[2]), list(args[3])))
                print("Simple")
                for i in range(len(args[0])):
                    print("{}: {}".format(i, list(args[0][i])))
                print("Nx {}".format(args[1].edges))
                return results

    if good:
        print("All good, matching results.")

    return results
