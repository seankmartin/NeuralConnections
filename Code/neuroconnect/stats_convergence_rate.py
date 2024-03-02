"""Analysing convergence rate of stats distributions."""

import os
import json
from configparser import ConfigParser
from types import SimpleNamespace
from collections import OrderedDict

import numpy as np
import pandas as pd
from mpmath import sqrt

from .connect_math import hypergeometric_pmf, create_uniform
from .monte_carlo import (
    monte_carlo,
    list_to_df,
    summarise_monte_carlo,
    get_distribution,
    dist_difference,
)
from .main import main as control_main
from .mpf_connection import CombProb
from .connect_math import expected_unique
from .connectivity_patterns import get_by_name
from .experiment import do_full_experiment


here = os.path.dirname(os.path.realpath(__file__))


def test_hyper_convergence_rate(N, K, n, num_iters=1000, num_cpus=1):
    """Compare simulated hypergeometric_pmf to actual over num_iters."""
    actual_distribution = {}
    expected = 0
    variance = 0
    for k in range(n + 1):
        actual_distribution[k] = hypergeometric_pmf(N, K, n, k)
        expected += actual_distribution[k] * k
        variance += actual_distribution[k] * k * k
    variance = variance - (expected * expected)

    total = np.array([i for i in range(N)])
    good = np.random.choice(total, size=K, replace=False)

    def random_var_gen(iter_val):
        drawn = np.random.choice(total, size=n, replace=False)
        return (drawn,)

    def fn_to_eval(drawn):
        count = 0
        for val in drawn:
            if val in good:
                count += 1
        return (count,)

    result = monte_carlo(fn_to_eval, random_var_gen, num_iters, num_cpus=num_cpus)
    df = list_to_df(
        result,
        [
            "Connections",
        ],
    )
    dist = get_distribution(df, "Connections", num_iters)
    os.makedirs(os.path.join(here, "..", "figures"), exist_ok=True)
    result = summarise_monte_carlo(
        df,
        to_plot=[
            "Connections",
        ],
        plt_outfile=os.path.join(here, "..", "figures", "dist.png"),
    )

    diff = dist_difference(actual_distribution, dist)
    return {
        "actual": actual_distribution,
        "simulated": dist,
        "difference": diff,
        "sim_summary": result,
        "stats_exp": (expected, sqrt(variance)),
    }


def test_network_convergence(num_cpus=1):
    """Test how fast the simulated networks converge to stability."""

    def delta_fn(**delta_params):
        N = delta_params.get("N")
        connections = delta_params.get("connections")
        total_samples = delta_params.get("total_samples")
        odict = OrderedDict()
        for i in range(total_samples + 1):
            odict[i] = OrderedDict()
            odict[i][int(expected_unique(N, i * connections))] = 1.0
        return odict

    cp = CombProb(1000, 50, 100, 1000, 50, delta_fn, True, connections=20, N=1000)

    vals = []
    result = {
        "expected": cp.expected_connections(),
        "total": cp.get_all_prob(),
    }
    for k, v in result["total"].items():
        vals.append([k, float(v), "Statistical estimation"])

    total = np.array([i for i in range(1000)])
    good = np.random.choice(total, size=100, replace=False)

    def random_var_gen(iter_val):
        drawn_a = np.random.choice(total, size=50, replace=False)
        count = 0
        for val in drawn_a:
            if val in good:
                count += 1

        good_b = np.random.choice(total, size=count * 20, replace=True)
        drawn_b = np.random.choice(total, size=50, replace=False)

        return (drawn_b, good_b)

    def fn_to_eval(drawn_b, good_b):
        count = 0
        for val in drawn_b:
            if val in good_b:
                count += 1
        return (count,)

    num_iters = 100000
    result = monte_carlo(fn_to_eval, random_var_gen, num_iters, num_cpus=num_cpus)
    df = list_to_df(
        result,
        [
            "Connections",
        ],
    )

    for n in [1000, 10000, 50000, num_iters]:
        dist = get_distribution(df.head(n), "Connections", n)
        for k, v in dist.items():
            vals.append([k, float(v), "Monte Carlo simulation {}".format(n)])

    columns = ["Number of sampled connected neurons", "Probability", "Calculation"]
    df = pd.DataFrame(vals, columns=columns)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "stats_convergence_fixed.csv"),
        index=False,
    )
    return df


def make_test_net(has_var=True):
    """Make a test network."""
    total = np.array([i for i in range(1000)])
    good = np.random.choice(total, size=200, replace=False)
    bad = []
    for val in total:
        if val not in good:
            bad.append(val)
    connections = {}

    if has_var:
        senders = np.random.choice(total, size=(200, 250), replace=True)
    else:
        senders = np.random.choice(total, size=(200, 150), replace=True)
    for i, val in enumerate(good):
        if has_var:
            num_choices = np.random.randint(50, 250 + 1, dtype=np.int32)
        else:
            num_choices = 150
        forward_connection = senders[i, :num_choices]
        connections[val] = forward_connection

    def random_var_gen(iter_val):
        f_arrays = []
        drawn_a = np.random.choice(total, size=20, replace=False)
        for val in drawn_a:
            if val in good:
                f_arrays.append(connections[val])
        if len(f_arrays) > 0:
            good_b = np.concatenate(f_arrays)
        else:
            good_b = []
        drawn_b = np.random.choice(total, size=20, replace=False)

        return (drawn_b, good_b)

    def fn_to_eval(drawn_b, good_b):
        count = 0
        for val in drawn_b:
            if val in good_b:
                count += 1
        return (count,)

    return random_var_gen, fn_to_eval


def test_rand_network_convergence(num_cpus=1, sr=None):
    """Test convergence of random networks."""
    rv = get_by_name("recurrent_connectivity")
    ndelta_fn = rv.static_expected_connections
    unif_out = create_uniform(50, 250)
    unif_re = create_uniform(50, 250)

    inter_dist = OrderedDict()
    inter_dist[0] = 1
    vals = []
    delta_params = {
        "out_connections_dist": unif_out,
        "recurrent_connections_dist": unif_re,
        "num_senders": 200,
        "num_recurrent": 0,
        "num_start": 1000,
        "total_samples": 20,
        "start_inter_dist": inter_dist,
        "end_inter_dist": inter_dist,
        "static_verbose": False,
        "max_depth": 1,
        "N": 1000,
    }

    cp = CombProb(1000, 20, 200, 1000, 20, ndelta_fn, subsample_rate=sr, **delta_params)

    mpf_result = {
        "expected": cp.expected_connections(),
        "total": cp.get_all_prob(),
    }
    expected = mpf_result["expected"]
    variance = 0
    for k, v in mpf_result["total"].items():
        vals.append([k, float(v), "Statistical estimation"])
        variance += k * k * v
    std = sqrt(variance - (expected * expected))
    print(mpf_result["expected"], std)

    rv = get_by_name("mean_connectivity")
    ndelta_fn = rv.static_expected_connections
    cp = CombProb(1000, 20, 200, 1000, 20, ndelta_fn, **delta_params)

    mpf_result = {
        "expected": cp.expected_connections(),
        "total": cp.get_all_prob(),
    }
    for k, v in mpf_result["total"].items():
        vals.append([k, float(v), "Mean estimation"])

    # Let's do 1000 iterations on 1000 graphs?
    # result = []
    # for j in tqdm.tqdm(range(50)):
    #     random_var_gen, fn_to_eval = make_test_net()
    #     r = monte_carlo(
    #         fn_to_eval, random_var_gen, 10000, num_cpus=num_cpus, progress=False
    #     )
    #     for val in r:
    #         result.append(val)
    # df = list_to_df(result, ["Connections",],)
    # dist = get_distribution(df, "Connections", 1000 * 100)
    # for k, v in dist.items():
    #     vals.append([k, float(v), "Monte Carlo simulation {}".format("50000")])

    random_var_gen, fn_to_eval = make_test_net()
    result = monte_carlo(fn_to_eval, random_var_gen, 50000, num_cpus=num_cpus)
    df = list_to_df(
        result,
        [
            "Connections",
        ],
    )

    for n in [1000, 100, 500]:
        expected = 0
        variance = 0
        dist = get_distribution(df.head(n), "Connections", n)
        for k, v in dist.items():
            vals.append([k, float(v), "Monte Carlo simulation {}".format(n)])
            expected += k * v
            variance += k * k * v
        std = sqrt(variance - (expected * expected))
        print(expected, std)

    columns = ["Number of sampled connected neurons", "Probability", "Calculation"]
    df = pd.DataFrame(vals, columns=columns)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "stats_convergence_rand.csv"),
        index=False,
    )

    return df


def test_config_convergence(config, out_name, num_cpus=1, max_depth=1):
    """Test convergence of network from a config file."""
    columns = ["Number of sampled connected neurons", "Probability", "Calculation"]
    vals = []
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
        num_iters=100000,
        max_depth=max_depth,
        gen_graph_each_iter=False,
        do_fixed=-1,
    )

    for k, v in result["mpf"]["total"].items():
        vals.append([k, float(v), "Statistical estimation"])

    df = result["graph"]["full_results"]
    for n in [1000, 10000, 50000, 100000]:
        dist = get_distribution(df.head(n), "Connections", n)
        for k, v in dist.items():
            vals.append([k, float(v), "Monte Carlo simulation {}".format(n)])

    result = do_full_experiment(
        region_sizes,
        get_by_name("mean_connectivity"),
        connectivity_params,
        num_samples,
        do_mpf,
        False,
        do_nx,
        do_vis_graph,
        num_iters=100000,
        max_depth=max_depth,
        gen_graph_each_iter=False,
        do_fixed=-1,
    )
    for k, v in result["mpf"]["total"].items():
        vals.append([k, float(v), "Mean estimation"])

    df = pd.DataFrame(vals, columns=columns)
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    df.to_csv(
        os.path.join(here, "..", "results", "convergence_{}.csv".format(out_name)),
        index=False,
    )

    return df


def main(N, K, n, num_iters):
    """Check convergence rate of hypergeometric_pmf and network dist."""
    np.random.seed(42)
    res1 = test_hyper_convergence_rate(N, K, n, num_iters)
    cfg_path = os.path.join(here, "..", "configs", "stats_check.cfg")
    cfg = ConfigParser()
    cfg.read(cfg_path)
    args = SimpleNamespace(max_depth=1, num_cpus=1, cfg="stats_check")
    print("Writing graph convergence to file in results directory")
    control_main(cfg, args)
    return res1


if __name__ == "__main__":
    np.random.seed(42)
    from pprint import pprint

    pprint(test_rand_network_convergence())
