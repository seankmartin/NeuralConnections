"""Compound operations to perform more complex tasks than control."""

import os
from configparser import ConfigParser
import json
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from skm_pyutils.config import print_cfg
from skm_pyutils.table import list_to_df, df_to_file, df_from_file
from dictances.bhattacharyya import bhattacharyya

from .monte_carlo import (
    get_distribution,
    monte_carlo,
    summarise_monte_carlo,
)
from .simple_graph import find_connected_limited, reverse, to_matrix
from .experiment import do_full_experiment
from .connectivity_patterns import OutgoingDistributionConnections, get_by_name
from .matrix import main as mouse_main
from .matrix import convert_mouse_data, load_matrix_data, matrix_vis
from .mpf_connection import CombProb
from .connect_math import create_uniform, discretised_rv
from .stored_results import store_mouse_result
from .atlas import (
    place_probes_at_com,
    get_n_random_points_in_region,
    get_brain_region_meshes,
    get_idx_of_points_in_meshes,
)
from .atlas_graph import prob_connect_probe


here = os.path.dirname(os.path.realpath(__file__))


def proportion(config, depths=[1, 2, 3]):
    """Load a config, change a var, and plot the result over the change."""
    np.random.seed(42)
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    region_sizes = json.loads(config.get("default", "region_sizes"))
    num_samples = json.loads(config.get("default", "num_samples"))
    num_iters = int(config.get("default", "num_iters"))
    connectivity_pattern = config.get("default", "connectivity_pattern")
    connectivity_pattern = get_by_name(connectivity_pattern)
    connectivity_param_names = json.loads(
        config.get("default", "connectivity_param_names")
    )

    connectivity_param_vals = []
    for name in connectivity_param_names:
        cfg_val = config.get("default", name, fallback=None)
        if cfg_val is not None:
            val = json.loads(cfg_val)
        else:
            val = [0 for _ in range(len(region_sizes))]
        connectivity_param_vals.append(val)

    connectivity_params = []
    for i in range(len(region_sizes)):
        d = {}
        for k, v in zip(connectivity_param_names, connectivity_param_vals):
            d[k] = v[i]
        connectivity_params.append(d)

    do_mpf = True
    do_graph = False
    do_nx = False
    do_vis_graph = False
    full_num = num_samples[0]
    samples = [i for i in range(1, num_samples[0] + 10)]

    for depth in depths:
        saved = []
        saved_least = []
        for val in samples:
            new_samples = [val] * 2
            result = do_full_experiment(
                region_sizes,
                connectivity_pattern,
                connectivity_params,
                new_samples,
                do_mpf,
                do_graph,
                do_nx,
                do_vis_graph,
                num_iters=num_iters,
                max_depth=depth,
            )
            at_least_one = 1 - result["mpf"]["total"][0]
            expected = result["mpf"]["expected"]
            proportion = expected / val
            saved.append(proportion)
            saved_least.append(at_least_one)

            if val == full_num:
                full = result["mpf"]

        here = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(os.path.join(here, "..", "figures"), exist_ok=True)

        fig, ax = plt.subplots()
        ax.plot(np.array(samples, dtype=float), np.array(saved, dtype=float), c="k")
        print("Setting x ticks")
        # plt.xticks([i for i in range(num_samples[0])])
        sns.despine()
        plt.xlabel("Number of recorded neurons")
        plt.ylabel("Proportion of connection receiving neurons")
        fig.savefig(
            os.path.join(here, "..", "figures", "proportion_{}.pdf".format(depth)),
            dpi=400,
        )

        fig, ax = plt.subplots()
        ax.plot(
            np.array(samples, dtype=float), np.array(saved_least, dtype=float), c="k"
        )
        sns.despine()
        plt.xlabel("Number of recorded neurons")
        plt.ylabel("Probability of at least one neuron receiving a connection")
        fig.savefig(
            os.path.join(here, "..", "figures", "prob_one_{}.pdf").format(depth),
            dpi=400,
        )

        fig, ax = plt.subplots()
        x = np.array(list(full["total"].keys()), dtype=float)
        y = np.array(list(full["total"].values()), dtype=float)

        ax.plot(x, y, "ko", ms=2.5)
        y_vals_min = [0 for _ in x]
        y_vals_max = y
        colors = ["k" for _ in x]
        ax.set_xticks([i for i in range(num_samples[0] + 1)])
        ax.set_xticklabels([i for i in range(num_samples[0] + 1)])
        # ax.set_ylim([0, 1])
        ax.vlines(x, y_vals_min, y_vals_max, colors=colors)
        sns.despine(offset=0, trim=True)
        plt.xlabel("Number of recorded connections")
        plt.ylabel("Probability")
        fig.savefig(
            os.path.join(here, "..", "figures", "pdf_samples_{}.pdf").format(depth),
            dpi=400,
        )


def pmf_accuracy(
    config,
    out_name,
    clt_start=10,
    sr=0.01,
    num_iters=50000,
    depth_full=3,
    num_graphs=1,
    do_the_stats=True,
):
    """Return pmf on samples."""
    np.random.seed(42)
    region_sizes = json.loads(config.get("default", "region_sizes"))
    num_samples = json.loads(config.get("default", "num_samples"))
    connectivity_pattern = config.get("default", "connectivity_pattern")
    connectivity_pattern = get_by_name(connectivity_pattern)
    connectivity_param_names = json.loads(
        config.get("default", "connectivity_param_names")
    )

    connectivity_param_vals = []
    for name in connectivity_param_names:
        cfg_val = config.get("default", name, fallback=None)
        if cfg_val is not None:
            val = json.loads(cfg_val)
        else:
            val = [0 for _ in range(len(region_sizes))]
        connectivity_param_vals.append(val)

    connectivity_params = []
    for i in range(len(region_sizes)):
        d = {}
        for k, v in zip(connectivity_param_names, connectivity_param_vals):
            d[k] = v[i]
        connectivity_params.append(d)

    do_mpf = True
    do_graph = True
    do_nx = False
    do_vis_graph = False

    vals = []
    mean_vals = []
    for max_depth in range(1, depth_full + 1):
        result = do_full_experiment(
            region_sizes,
            connectivity_pattern,
            connectivity_params,
            num_samples,
            do_mpf,
            False,
            do_nx,
            do_vis_graph,
            num_iters=num_iters,
            max_depth=max_depth,
            clt_start=clt_start,
            subsample_rate=sr,
            use_mean=do_the_stats,
        )

        cols = ["Number of connected neurons", "Probability", "Calculation"]
        mpf_res = result["mpf"]["total"]

        for k, v in mpf_res.items():
            vals.append([k, v, "Statistical estimation {}".format(max_depth)])

        mean_vals.append(
            [max_depth, "Statistical estimation", float(result["mpf"]["expected"])]
        )

        mean = 0
        for i in range(num_graphs):
            result = do_full_experiment(
                region_sizes,
                connectivity_pattern,
                connectivity_params,
                num_samples,
                False,
                do_graph,
                do_nx,
                do_vis_graph,
                num_iters=num_iters,
                max_depth=max_depth,
                quiet=True,
            )
            graph_res = result["graph"]["dist"]

            for k, v in graph_res.items():
                vals.append([k, v, "Monte Carlo simulation {}".format(max_depth)])
            mean += np.mean(result["graph"]["full_results"]["Connections"].values)

        mean = mean / num_graphs
        mean_vals.append([max_depth, "Monte Carlo simulation", mean])

        result = do_full_experiment(
            region_sizes,
            get_by_name("mean_connectivity"),
            connectivity_params,
            num_samples,
            do_mpf,
            False,
            False,
            False,
            num_iters=num_iters,
            max_depth=max_depth,
            subsample_rate=None,
        )
        mpf_res = result["mpf"]["total"]

        for k, v in mpf_res.items():
            vals.append([k, v, "Mean estimation {}".format(max_depth)])

        mean_vals.append(
            [max_depth, "Mean estimation", float(result["mpf"]["expected"])]
        )

    df = pd.DataFrame(vals, columns=cols)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "pmf_comp_{}.csv".format(out_name)),
        index=False,
    )

    mean_cols = ["Depth", "Calculation", "Mean"]
    df = pd.DataFrame(mean_vals, columns=mean_cols)
    df.to_csv(
        os.path.join(here, "..", "results", "pmf_mean_{}.csv".format(out_name)),
        index=False,
    )

    return df


def connections_dependent_on_samples(
    config,
    out_name,
    do_graph=True,
    num_iters=15000,
    use_mean=True,
    num_graphs=1,
    sr=0.01,
    clt_start=30,
    fin_depth=3,
):
    """Return expected connections on samples."""
    np.random.seed(42)
    region_sizes = json.loads(config.get("default", "region_sizes"))
    num_samples = json.loads(config.get("default", "num_samples"))

    connectivity_pattern = config.get("default", "connectivity_pattern")
    connectivity_pattern = get_by_name(connectivity_pattern)
    connectivity_param_names = json.loads(
        config.get("default", "connectivity_param_names")
    )

    connectivity_param_vals = []
    for name in connectivity_param_names:
        cfg_val = config.get("default", name, fallback=None)
        if cfg_val is not None:
            val = json.loads(cfg_val)
        else:
            val = [0 for _ in range(len(region_sizes))]
        connectivity_param_vals.append(val)

    connectivity_params = []
    for i in range(len(region_sizes)):
        d = {}
        for k, v in zip(connectivity_param_names, connectivity_param_vals):
            d[k] = v[i]
        connectivity_params.append(d)

    do_mpf = True
    do_nx = False
    do_vis_graph = False

    vals = []
    cols = [
        "Number of samples",
        "Expected connected",
        "Expected proportion connected",
        "Max distance",
        "Calculation",
    ]
    depth_name = [None, "Direct synapse", "Two synapses", "Three synapses"]
    for max_depth in range(1, fin_depth + 1):
        samples = [i for i in range(num_samples[0] + 1)]
        for val in samples:
            if val == 0:
                vals.append(
                    [val, 0, 0, depth_name[max_depth], "Statistical estimation"]
                )
            else:
                new_samples = [val] * 2
                result = do_full_experiment(
                    region_sizes,
                    connectivity_pattern,
                    connectivity_params,
                    new_samples,
                    do_mpf,
                    False,
                    do_nx,
                    do_vis_graph,
                    num_iters=num_iters,
                    max_depth=max_depth,
                    save_every=1,
                    subsample_rate=sr,
                    use_mean=use_mean,
                    clt_start=clt_start,
                )
                vals.append(
                    [
                        val,
                        result["mpf"]["expected"],
                        result["mpf"]["expected"] / val,
                        depth_name[max_depth],
                        "Statistical estimation",
                    ]
                )
            if do_graph:
                if val == 0:
                    vals.append(
                        [val, 0, 0, depth_name[max_depth], "Monte carlo simulation"]
                    )
                else:
                    for _ in range(num_graphs):
                        result = do_full_experiment(
                            region_sizes,
                            connectivity_pattern,
                            connectivity_params,
                            new_samples,
                            False,
                            do_graph,
                            do_nx,
                            do_vis_graph,
                            num_iters=num_iters,
                            max_depth=max_depth,
                            save_every=1,
                            use_mean=use_mean,
                            quiet=True,
                        )
                        connects = result["graph"]["full_results"]["Connections"].values
                        for df_val in connects:
                            vals.append(
                                [
                                    val,
                                    df_val,
                                    df_val / val,
                                    depth_name[max_depth],
                                    "Monte carlo simulation",
                                ]
                            )
    df = pd.DataFrame(vals, columns=cols)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(
            here, "..", "results", "connection_samples_{}.csv".format(out_name)
        ),
        index=False,
    )

    return df


def connections_dependent_on_regions(
    cfg_names,
    r_names,
    depths,
    out_name,
    num_iters=20000,
    do_graph=True,
):
    """Return connection expectation for different regions or connectivity."""
    vals = []
    cols = [
        "Expected connected",
        "Expected proportion connected",
        "Connectivity",
        "Calculation",
    ]

    for cfg_name, r_name, max_depth in zip(cfg_names, r_names, depths):
        here = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(here, "..", "configs", cfg_name)
        config = ConfigParser()
        config.read(config_path)

        region_sizes = json.loads(config.get("default", "region_sizes"))
        num_samples = json.loads(config.get("default", "num_samples"))
        connectivity_pattern = config.get("default", "connectivity_pattern")
        connectivity_pattern = get_by_name(connectivity_pattern)
        connectivity_param_names = json.loads(
            config.get("default", "connectivity_param_names")
        )

        connectivity_param_vals = []
        for name in connectivity_param_names:
            cfg_val = config.get("default", name, fallback=None)
            if cfg_val is not None:
                val = json.loads(cfg_val)
            else:
                val = [0 for _ in range(len(region_sizes))]
            connectivity_param_vals.append(val)

        connectivity_params = []
        for i in range(len(region_sizes)):
            d = {}
            for k, v in zip(connectivity_param_names, connectivity_param_vals):
                d[k] = v[i]
            connectivity_params.append(d)

        do_mpf = True
        do_nx = False
        do_vis_graph = False

        result = do_full_experiment(
            region_sizes,
            connectivity_pattern,
            connectivity_params,
            num_samples,
            do_mpf,
            do_graph,
            do_nx,
            do_vis_graph,
            num_iters=num_iters,
            max_depth=max_depth,
            gen_graph_each_iter=False,
        )

        vals.append(
            [
                result["mpf"]["expected"],
                result["mpf"]["expected"] / num_samples[1],
                r_name,
                "Statistical estimation",
            ]
        )

        if do_graph:
            to_add = np.mean(result["graph"]["full_results"]["Connections"].values)
            vals.append(
                [
                    to_add,
                    to_add / num_samples[1],
                    r_name,
                    "Monte Carlo simulation",
                ]
            )

    df = pd.DataFrame(vals, columns=cols)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "region_exp_{}.csv".format(out_name)),
        index=False,
    )

    return df


def distance_dependent_on_regions(
    cfg_names,
    r_names,
    depths,
    out_name,
    num_iters=20000,
):
    """Return connection expectation for different regions or connectivity."""
    np.random.seed(42)
    vals = []
    cols = [
        "Bhattacharyya distance",
        "Connectivity",
    ]

    for cfg_name, r_name, max_depth in zip(cfg_names, r_names, depths):
        if cfg_name == "USE STORED MOUSE":
            dist = store_mouse_result()
            vals.append([dist, r_name])
            continue
        here = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(here, "..", "configs", cfg_name)
        config = ConfigParser()
        config.read(config_path)

        region_sizes = json.loads(config.get("default", "region_sizes"))
        num_samples = json.loads(config.get("default", "num_samples"))
        connectivity_pattern = config.get("default", "connectivity_pattern")
        connectivity_pattern = get_by_name(connectivity_pattern)
        connectivity_param_names = json.loads(
            config.get("default", "connectivity_param_names")
        )

        connectivity_param_vals = []
        for name in connectivity_param_names:
            cfg_val = config.get("default", name, fallback=None)
            if cfg_val is not None:
                val = json.loads(cfg_val)
            else:
                val = [0 for _ in range(len(region_sizes))]
            connectivity_param_vals.append(val)

        connectivity_params = []
        for i in range(len(region_sizes)):
            d = {}
            for k, v in zip(connectivity_param_names, connectivity_param_vals):
                d[k] = v[i]
            connectivity_params.append(d)

        do_mpf = True
        do_graph = True
        do_nx = False
        do_vis_graph = False

        result = do_full_experiment(
            region_sizes,
            connectivity_pattern,
            connectivity_params,
            num_samples,
            do_mpf,
            do_graph,
            do_nx,
            do_vis_graph,
            num_iters=num_iters,
            max_depth=max_depth,
            gen_graph_each_iter=False,
        )

        dist_estimate = result["mpf"]["total"]
        dist_actual = result["graph"]["dist"]
        distance = bhattacharyya(dist_estimate, dist_actual)
        vals.append([distance, r_name])

    df = pd.DataFrame(vals, columns=cols)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "region_bhatt_{}.csv".format(out_name)),
        index=False,
    )

    return df


def mouse_region_exp(
    regions, depths, out_name, num_samples, num_iters=1000, do_graph=False
):
    """The expected value from different mouse brain regions."""
    np.random.seed(42)
    vals = []
    cols = [
        "Expected connected",
        "Expected proportion connected",
        "Regions",
        "Calculation",
    ]

    for r, d in zip(regions, depths):
        result = mouse_main(
            num_sampled=num_samples,
            max_depth=d,
            num_iters=num_iters,
            do_graph=do_graph,
            only_exp=True,
            A_name=r[0],
            B_name=r[1],
        )
        vals.append(result[0])
        if result[1] is not None:
            vals.append(result[1])

    df = pd.DataFrame(vals, columns=cols)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "mouse_region_exp_{}.csv".format(out_name)),
        index=False,
    )

    return df


def mouse_region_exp_probes(
    regions,
    num_sampled,
    colors=None,
    style="cartoon",
    interactive=False,
    hemisphere="right",
    vis_only=False,
    block_size_sub=10,
    probe_kwargs=None,
    vis_full=False,
    **simulation_kwargs,
):
    """The expected value from different mouse brain regions with probes."""
    np.random.seed(42)

    cols = ["Number of connected neurons", "Probability", "Calculation", "Regions"]

    if probe_kwargs is None:
        probe_kwargs = [None] * len(regions)

    for r, pk in zip(regions, probe_kwargs):
        A_name, B_name = r
        final_res_list = []

        name = f"{r[0]}_to_{r[1]}_render"
        cylinder = place_probes_at_com(
            r,
            hemisphere=hemisphere,
            colors=colors,
            style=style,
            interactive=interactive,
            screenshot_name=name,
            probe_kwargs=pk,
        )
        if vis_only:
            continue

        # Load mouse data
        convert_mouse_data(A_name, B_name)
        to_use = [True, True, True, True]
        mc, full_stats = load_matrix_data(to_use, A_name, B_name, hemisphere=hemisphere)
        print("{} - {}, {} - {}".format(A_name, B_name, mc.num_a, mc.num_b))
        region_sizes = [mc.num_a, mc.num_b]

        if (A_name == "MOp" and B_name == "SSp-ll") or vis_full:
            o_name = "mc_mat_vis_{}_to_{}.pdf".format(A_name, B_name)
            print("Plotting full matrix vis")
            matrix_vis(mc.ab, mc.ba, mc.aa, mc.bb, 150, name=o_name)

        t = time.perf_counter()
        print("Creating graph")
        mc.create_connections()
        t2 = time.perf_counter() - t
        print(f"Finished graph creation in {t2:.2f}s")

        # Find intersections of probes and cells
        brain_region_meshes = get_brain_region_meshes(r, None, hemisphere=hemisphere)

        t = time.perf_counter()
        print("Placing cells in device")
        region_pts = []
        for region_mesh, region_size in zip(brain_region_meshes, region_sizes):
            pts = get_n_random_points_in_region(region_mesh, region_size, sort_=True)
            meshes = [cylinder]
            pts_idxs = np.sort(get_idx_of_points_in_meshes(pts, meshes))
            pts = pts[pts_idxs]
            region_pts.append((pts, pts_idxs))
        t2 = time.perf_counter() - t
        print(f"Finished cells creation in {t2:.2f}s")

        a_indices = region_pts[0][1]
        b_indices = region_pts[1][1]

        t = time.perf_counter()
        print("Visualsing matrix")
        mc_sub = mc.subsample(a_indices, b_indices)
        o_name = f"{r[0]}_to_{r[1]}_connection_matrix_subbed.pdf"
        matrix_vis(mc_sub.ab, mc_sub.ba, mc_sub.aa, mc_sub.bb, block_size_sub, o_name)
        t2 = time.perf_counter() - t
        print(f"Finished matrix vis in {t2:.2f}s")

        # Probability calculation
        t = time.perf_counter()
        print("Running simulation")
        res = prob_connect_probe(
            mc, num_sampled, a_indices, b_indices, full_stats, **simulation_kwargs
        )
        t2 = time.perf_counter() - t
        print(f"Finished simulation in {t2:.2f}s")

        r_str = f"{r[0]}_{r[1]}"
        for k, v in res[0]["dist"].items():
            final_res_list.append([k, v, "Monte Carlo simulation", r_str])

        for k, v in res[1]["total"].items():
            final_res_list.append([k, v, "Statistical estimation", r_str])

        max_depth = simulation_kwargs["max_depth"]
        df = list_to_df(final_res_list, headers=cols)
        fname = f"sub_regions_{r[0]}_{r[1]}_depth_{max_depth}.csv"
        fname = os.path.join(here, "..", "results", fname)
        print("Saved dataframe results to {}".format(fname))
        df_to_file(df, fname, index=False)

    l = []
    for r in regions:
        max_depth = simulation_kwargs["max_depth"]
        fname = f"sub_regions_{r[0]}_{r[1]}_depth_{max_depth}.csv"
        fname = os.path.join(here, "..", "results", fname)
        df = df_from_file(fname)

        for calculation in ["Monte Carlo simulation", "Statistical estimation"]:
            df_stats = df[df[cols[2]] == calculation]
            expected = (df_stats[cols[0]] * df_stats[cols[1]]).sum()
            if num_sampled[1] == 0:
                expected_proportion = 0
            else:
                expected_proportion = expected / num_sampled[1]
            regions = df[cols[-1]][0]
            l.append([expected, expected_proportion, regions, calculation])

    cols = [
        "Expected connected",
        "Expected proportion connected",
        "Regions",
        "Calculation",
    ]
    new_df = list_to_df(l, headers=cols)
    new_df.to_csv(
        os.path.join(here, "..", "results", "mouse_region_exp_probes.csv"),
        index=False,
    )

    return new_df


def mouse_region_depths(
    regions,
    num_sampled,
    colors=None,
    style="cartoon",
    interactive=False,
    hemisphere="right",
    vis_only=False,
    probe_kwargs=None,
    **simulation_kwargs,
):
    np.random.seed(42)

    cols = [
        "Proportion of connections",
        "Regions",
        "Max distance",
        "Number of samples",
    ]

    if probe_kwargs is None:
        probe_kwargs = [None] * len(regions)

    for r, pk in zip(regions, probe_kwargs):
        final_res_list = []

        name = f"{r[0]}_to_{r[1]}_render"
        cylinder = place_probes_at_com(
            r,
            hemisphere=hemisphere,
            colors=colors,
            style=style,
            interactive=interactive,
            screenshot_name=name,
            probe_kwargs=pk,
        )
        if vis_only:
            continue

        # Load mouse data
        A_name, B_name = r
        convert_mouse_data(A_name, B_name)
        to_use = [True, True, True, True]
        mc, full_stats = load_matrix_data(to_use, A_name, B_name, hemisphere=hemisphere)
        print("{} - {}, {} - {}".format(A_name, B_name, mc.num_a, mc.num_b))
        region_sizes = [mc.num_a, mc.num_b]

        # Find intersections of probes and cells
        brain_region_meshes = get_brain_region_meshes(r, None, hemisphere=hemisphere)

        region_pts = []
        for region_mesh, region_size in zip(brain_region_meshes, region_sizes):
            pts = get_n_random_points_in_region(region_mesh, region_size, sort_=True)
            meshes = [cylinder]
            pts_idxs = np.sort(get_idx_of_points_in_meshes(pts, meshes))
            pts = pts[pts_idxs]
            region_pts.append((pts, pts_idxs))

        a_indices = region_pts[0][1]
        b_indices = region_pts[1][1]

        # Probability calculation

        for ns in range(num_sampled + 1):
            for depth in range(1, 4):
                simulation_kwargs["max_depth"] = depth
                res = prob_connect_probe(
                    mc,
                    [ns, ns],
                    a_indices,
                    b_indices,
                    full_stats,
                    do_graph=False,
                    **simulation_kwargs,
                )
                r_str = f"{r[0]}_{r[1]}"
                exp = res[1]["expected"]
                if ns != 0:
                    final_res_list.append([exp / ns, r_str, depth, ns])
                else:
                    final_res_list.append([exp, r_str, depth, ns])

        df = list_to_df(final_res_list, headers=cols)
        fname = f"sub_regions_{r[0]}_{r[1]}_all_depth.csv"
        fname = os.path.join(here, "..", "results", fname)
        print("Saved dataframe results to {}".format(fname))
        df_to_file(df, fname, index=False)


def out_exp(config, out_name, depth, num_iters=1000):
    """The expected number of receiving neurons in region 2."""
    np.random.seed(42)
    vals = []
    cols = [
        "Number of samples",
        "Expected connected",
        "Expected proportion connected",
        "Calculation",
    ]

    region_sizes = json.loads(config.get("default", "region_sizes"))
    num_samples = json.loads(config.get("default", "num_samples"))
    connectivity_pattern = config.get("default", "connectivity_pattern")
    connectivity_pattern = get_by_name(connectivity_pattern)
    connectivity_param_names = json.loads(
        config.get("default", "connectivity_param_names")
    )

    connectivity_param_vals = []
    for name in connectivity_param_names:
        cfg_val = config.get("default", name, fallback=None)
        if cfg_val is not None:
            val = json.loads(cfg_val)
        else:
            val = [0 for _ in range(len(region_sizes))]
        connectivity_param_vals.append(val)

    connectivity_params = []
    for i in range(len(region_sizes)):
        d = {}
        for k, v in zip(connectivity_param_names, connectivity_param_vals):
            d[k] = v[i]
        connectivity_params.append(d)

    do_mpf = True
    do_nx = False
    do_graph = True
    do_vis_graph = False

    result = do_full_experiment(
        region_sizes,
        connectivity_pattern,
        connectivity_params,
        num_samples,
        do_mpf,
        do_graph,
        do_nx,
        do_vis_graph,
        num_iters=num_iters,
        max_depth=depth,
        gen_graph_each_iter=False,
        do_fixed=1,
    )

    for k, v in result["mpf"]["each_expected"].items():
        vals.append([k, float(v), float(v) / region_sizes[1], "Statistical estimation"])

    for i in range(num_samples[0] + 1):
        to_add = result["g_{}".format(i)]["full_results"]["Connections"].values
        for v in to_add:
            vals.append([i, v, v / region_sizes[1], "Monte Carlo simulation"])

    df = pd.DataFrame(vals, columns=cols)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "total_b_exp_{}.csv".format(out_name)),
        index=False,
    )

    return df


def df_from_dict(dict, cols):
    """Form a dataframe from a dictionary with cols, keys are considered an entry."""
    vals = []
    for k, v in dict.items():
        vals.append([k, v])

    df = pd.DataFrame(vals, columns=cols)

    return df


def explain_calc(config, out_name="explain", sr=0.01):
    """Figures to explain how the calculation is performed."""
    np.random.seed(42)
    region_sizes = json.loads(config.get("default", "region_sizes"))
    num_samples = json.loads(config.get("default", "num_samples"))
    connectivity_pattern = config.get("default", "connectivity_pattern")
    connectivity_pattern = get_by_name(connectivity_pattern)
    connectivity_param_names = json.loads(
        config.get("default", "connectivity_param_names")
    )

    connectivity_param_vals = []
    for name in connectivity_param_names:
        cfg_val = config.get("default", name, fallback=None)
        if cfg_val is not None:
            val = json.loads(cfg_val)
        else:
            val = [0 for _ in range(len(region_sizes))]
        connectivity_param_vals.append(val)

    connectivity_params = []
    for i in range(len(region_sizes)):
        d = {}
        for k, v in zip(connectivity_param_names, connectivity_param_vals):
            d[k] = v[i]
        connectivity_params.append(d)

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
        "total_samples": num_samples[0],
        "start_inter_dist": inter_a,
        "end_inter_dist": inter_b,
        "static_verbose": False,
        "max_depth": 1,
        "init_delta": True,
        "clt_start": 30,
    }

    # Firstly, the results for a senders figure
    cp = CombProb(
        region_sizes[0],
        num_samples[0],
        connectivity_params[0]["num_senders"],
        region_sizes[1],
        num_samples[1],
        connectivity_pattern.static_expected_connections,
        cache=True,
        subsample_rate=sr,
        N=region_sizes[1],
        verbose=False,
        **delta_params,
    )

    data = cp.calculate_distribution_n_senders()
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df = df_from_dict(data, cols=["Number of sampled senders", "Probability"])
    df.to_csv(
        os.path.join(here, "..", "results", "a_prob_{}.csv".format(out_name)),
        index=False,
    )

    # Secondly, the results for a receivers figure
    data = cp.a_to_b_dist
    df = df_from_dict(data, cols=["Number of receivers", "Probability"])
    df.to_csv(
        os.path.join(here, "..", "results", "b_prob_{}.csv".format(out_name)),
        index=False,
    )

    # Thirdly, the probability of X=x for just one
    data = cp.final_distribution(keep_all=True)
    df = df_from_dict(data, cols=["Number of sampled senders", "Probability"])
    vals = []
    cols = ["Number of sampled A", "Number of receivers", "Probability"]
    for k, v in data.items():
        for k2, v2 in v.items():
            vals.append([k, k2, v2])
    df = pd.DataFrame(vals, columns=cols)
    df.to_csv(
        os.path.join(here, "..", "results", "b_each_{}.csv".format(out_name)),
        index=False,
    )

    # Lastly, the full marginal distribution
    data = cp.stored
    df = df_from_dict(data, cols=["Number of sampled receivers", "Probability"])
    df.to_csv(
        os.path.join(here, "..", "results", "b_fin_{}.csv".format(out_name)),
        index=False,
    )


def compare_distribution(dist, **kwargs):
    """
    Compare the accuracy of stats vs monte carlo on distributions.

    The idea is try multiple distributions for forward connections
    between two brain areas, and see how well the stats matches the simulations
    with different kinds of distributions.

    Parameters
    ----------
    dist : OrderedDict
        Discretised distribution

    Keyword arguments
    -----------------
    max_outgoing_connections : int
    region1_nodes : list
    region2_nodes : list
    num_region1_senders : int
    num_samples : int
    subsample_rate : float
    clt_start : int
    num_monte_carlo_iters : int
    do_matrix_visualisation : bool
    smoothing_win_size : int
    name : string

    """
    np.random.seed(42)
    region1_nodes = kwargs.get("region1_nodes")
    region2_nodes = kwargs.get("region2_nodes")
    num_region1_senders = kwargs.get("num_region1_senders")
    num_samples = kwargs.get("num_samples")
    subsample_rate = kwargs.get("subsample_rate")
    clt_start = kwargs.get("clt_start")
    num_monte_carlo_iters = kwargs.get("num_monte_carlo_iters")
    do_matrix_visualisation = kwargs.get("do_matrix_visualisation")
    smoothing_win_size = kwargs.get("smoothing_win_size")
    name = kwargs.get("name")

    result = {}
    connection_instance = OutgoingDistributionConnections(
        region1_nodes, region2_nodes, dist, num_region1_senders
    )

    def do_mpf():
        delta_params = dict(
            num_start=len(region1_nodes),
            num_end=len(region2_nodes),
            num_senders=num_region1_senders,
            out_connections_dist=dist,
            total_samples=num_samples[0],
            clt_start=clt_start,
            sub=subsample_rate,
        )
        connection_prob = CombProb(
            len(region1_nodes),
            num_samples[0],
            num_region1_senders,
            len(region2_nodes),
            num_samples[1],
            OutgoingDistributionConnections.static_expected_connections,
            subsample_rate=subsample_rate,
            approx_hypergeo=False,
            **delta_params,
        )
        result["mpf"] = {
            "expected": connection_prob.expected_connections(),
            "total": connection_prob.get_all_prob(),
            "each_expected": {
                k: connection_prob.expected_total(k) for k in range(num_samples[0] + 1)
            },
        }

    def do_graph():
        g_graph, g_connected = connection_instance.create_connections()
        g_graph.extend([[] for _ in region2_nodes])
        g_reverse_graph = reverse(g_graph)

        def random_var_gen(iter_val):
            graph, connected = g_graph, g_connected
            sources = np.random.choice(region1_nodes, num_samples[0], replace=False)
            targets = np.random.choice(region2_nodes, num_samples[-1], replace=False)

            return graph, sources, targets

        def fn_to_eval(graph, sources, targets):
            reverse_graph = g_reverse_graph
            reachable = find_connected_limited(
                graph, sources, targets, max_depth=1, reverse_graph=reverse_graph
            )
            return (len(reachable),)

        mc_res = monte_carlo(
            fn_to_eval,
            random_var_gen,
            num_monte_carlo_iters,
            num_cpus=1,
            headers=["Connections"],
            save_name="graph_mc.csv",
            save_every=10000,
            progress=True,
        )
        df = list_to_df(mc_res, ["Connections"])
        mc_res = summarise_monte_carlo(
            df, to_plot=["Connections"], plt_outfile="graph_dist.png"
        )
        distrib = get_distribution(df, "Connections", num_monte_carlo_iters)

        if do_matrix_visualisation:
            graph, _, _ = random_var_gen(0)
            AB, BA, AA, BB = to_matrix(graph, len(region1_nodes), len(region2_nodes))
            matrix_vis(
                AB,
                None,
                None,
                None,
                k_size=smoothing_win_size,
                name=f"{name}_graph_mat_vis.pdf",
            )

        result["graph"] = {"full_results": df, "summary_stats": mc_res, "dist": distrib}

    def save_results():
        dist_list = []
        mpf_res = result["mpf"]["total"]
        graph_res = result["graph"]["dist"]

        for k, v in mpf_res.items():
            v2 = graph_res.get(k, 0)
            dist_list.append([k, v, "Statistical estimation"])
            dist_list.append([k, v2, "Monte Carlo simulation"])

        cols = [
            "Number of recorded connected neurons",
            "Probability",
            "Calculation",
        ]
        df = list_to_df(dist_list, headers=cols)
        os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
        df.to_csv(os.path.join(here, "..", "results", f"{name}_accuracy.csv"))

    do_mpf()
    do_graph()
    save_results()


def main(cfg_name):
    """Calculate the proportion for the given config."""
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "..", "configs", cfg_name)
    cfg = ConfigParser()
    cfg.read(config_path)
    print_cfg(cfg, "Program started with configuration")
    sns.set_palette("colorblind")
    proportion(cfg)
