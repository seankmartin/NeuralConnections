"""Main function."""

import json
import os
from pprint import pprint

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .experiment import do_full_experiment
from .connectivity_patterns import get_by_name
from .monte_carlo import dist_difference


def main(config, args, already_parsed=False):
    """Perform a full experiment for a given config."""
    np.random.seed(42)

    here = os.path.dirname(os.path.realpath(__file__))
    if not already_parsed:
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

        do_mpf = config.getboolean("Setup", "do_mpf")
        do_graph = config.getboolean("Setup", "do_graph")
        do_nx = config.getboolean("Setup", "do_nx")
        do_vis_graph = config.getboolean("Setup", "do_vis_graph")
        do_only_none = config.getboolean("Setup", "do_only_none")
        do_fixed = int(config.get("Setup", "do_fixed", fallback=-1))
        gen_graph_each_iter = config.getboolean(
            "Setup", "gen_graph_each_iter", fallback=False
        )
        do_mat_vis = config.getboolean("Setup", "do_mat_vis", fallback=False)
        save_dir = os.path.join(here, "..", "results")
    else:
        region_sizes = config["default"]["region_sizes"]
        num_samples = config["default"]["num_samples"]
        num_iters = config["default"]["num_iters"]
        connectivity_pattern = config["default"]["connectivity_pattern"]
        connectivity_pattern = get_by_name(connectivity_pattern)
        connectivity_param_names = config["default"]["connectivity_param_names"]

        connectivity_param_vals = []
        for name in connectivity_param_names:
            val = config["default"][name]
            connectivity_param_vals.append(val)

        connectivity_params = []
        for i in range(len(region_sizes)):
            d = {}
            for k, v in zip(connectivity_param_names, connectivity_param_vals):
                d[k] = v[i]
            connectivity_params.append(d)

        do_mpf = config["Setup"]["do_mpf"]
        do_graph = config["Setup"]["do_graph"]
        do_nx = config["Setup"]["do_nx"]
        do_vis_graph = config["Setup"]["do_vis_graph"]
        do_only_none = config["Setup"]["do_only_none"]
        do_fixed = -1
        gen_graph_each_iter = config["Setup"]["gen_graph_each_iter"]
        do_mat_vis = False
        save_dir = config["Setup"]["save_dir"]

    result = do_full_experiment(
        region_sizes,
        connectivity_pattern,
        connectivity_params,
        num_samples,
        do_mpf,
        do_graph,
        do_nx,
        do_vis_graph,
        do_only_none,
        num_iters=num_iters,
        num_cpus=args.num_cpus,
        max_depth=args.max_depth,
        do_fixed=do_fixed,
        gen_graph_each_iter=gen_graph_each_iter,
        do_mat_vis=do_mat_vis,
        clt_start=args.clt_start,
        subsample_rate=args.subsample_rate,
        approx_hypergeo=args.approx_hypergeo,
    )
    if ("graph" in result.keys()) and ("mpf" in result.keys()):
        result["difference"] = dist_difference(
            result["mpf"]["total"], result["graph"]["dist"]
        )
    os.makedirs(os.path.join(here, "..", "results"), exist_ok=True)
    with open(
        os.path.join(
            save_dir,
            os.path.splitext(args.cfg)[0] + "_" + str(args.max_depth) + ".txt",
        ),
        "w",
    ) as f:
        print("------------------------CONFIG----------------------", file=f)
        print(args, file=f)
        print("", file=f)
        if not already_parsed:
            config_dict = [{x: tuple(config.items(x))} for x in config.sections()]
        else:
            config_dict = config
        pprint(config_dict, width=120, stream=f)
        print("", file=f)
        print("------------------------RESULT----------------------", file=f)
        pprint(result, width=120, stream=f)

    fig, ax = plt.subplots()
    if "mpf" in result.keys():
        mpf_res = result["mpf"]["total"]
        x = list(mpf_res.keys())
        y = list(mpf_res.values())
        ax.plot(x, y, c="k", label="Statistical estimation")
    if "graph" in result.keys():
        mpf_res = result["graph"]["dist"]
        x = list(mpf_res.keys())
        y = list(mpf_res.values())
        ax.plot(x, y, c="b", linestyle="--", label="Monte Carlo simulation")
    plt.legend()
    fig.savefig(
        os.path.join(
            save_dir,
            os.path.splitext(args.cfg)[0] + "_" + str(args.max_depth) + ".pdf",
        )
    )
    plt.close(fig)
    return result
