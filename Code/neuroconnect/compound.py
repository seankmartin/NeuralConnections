"""Compound operations to perform more complex tasks than control."""

import os
from configparser import ConfigParser
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .experiment import do_full_experiment
from .connectivity_patterns import get_by_name
from .matrix import main as mouse_main
from .mpf_connection import CombProb
from .connect_math import create_uniform
from .stored_results import store_mouse_result
from skm_pyutils.py_config import print_cfg
from dictances.bhattacharyya import bhattacharyya

here = os.path.dirname(os.path.realpath(__file__))


def proportion(config, depths=[1, 2, 3]):
    """Load a config, change a var, and plot the result over the change."""
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
    cfg_names, r_names, depths, out_name, num_iters=20000, do_graph=True,
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
                [to_add, to_add / num_samples[1], r_name, "Monte Carlo simulation",]
            )

    df = pd.DataFrame(vals, columns=cols)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "region_exp_{}.csv".format(out_name)),
        index=False,
    )

    return df


def distance_dependent_on_regions(
    cfg_names, r_names, depths, out_name, num_iters=20000,
):
    """Return connection expectation for different regions or connectivity."""
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
        print(dist_estimate, dist_actual)
        print(distance)
        print("-------------")
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


def out_exp(config, out_name, depth, num_iters=1000):
    """The expected number of receiving neurons in region 2."""
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
        **delta_params
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


def main(cfg_name):
    """Calculate the proportion for the given config."""
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "..", "configs", cfg_name)
    cfg = ConfigParser()
    cfg.read(config_path)
    print_cfg(cfg, "Program started with configuration")
    sns.set_palette("colorblind")
    proportion(cfg)
