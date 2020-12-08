"""A module to store some results that are parsed from .txt files."""

import os
from configparser import ConfigParser
from types import SimpleNamespace

import pandas as pd

from .main import main as ctrl_main

here = os.path.dirname(os.path.abspath(__file__))


def parse_cfg(name):
    """Parse the configs at configs/name."""
    cfg_path = os.path.join(here, "..", "configs", name)
    cfg = ConfigParser()
    cfg.read(cfg_path)

    return cfg


def df_from_dict(dict, cols):
    """Form a dataframe from a dictionary with cols, keys are considered an entry."""
    vals = []
    for k, v in dict.items():
        vals.append([k, v])

    df = pd.DataFrame(vals, columns=cols)

    return df


def store_region_results():
    vals = []
    names = [
        "Tetrode CA3 CA1",
        "MOp to SSp-ll",
        "Figure 1 F",
        "Max distance 3",
        "Figure 1 A",
    ]
    mean_vals = [
        0.4248 / 5.0,
        6.7371 / 79.0,
        8.512 / 20.0,
        8.86 / 25.0,
        0.7340 / 3.0,
    ]
    stats_vals = [
        0.4117 / 5.0,
        6.20478 / 79.0,
        8.511 / 20.0,
        9.27 / 25.0,
        0.7346 / 3.0,
    ]
    for i in range(len(names)):
        vals.append([names[i], mean_vals[i], "Monte Carlo simulation"])
        vals.append([names[i], stats_vals[i], "Statistical estimation"])

    cols = ["Connectivity", "Expected proportion connected", "Calculation"]
    df = pd.DataFrame(vals, columns=cols)
    df.to_csv(
        os.path.join(here, "..", "results", "exp_man.csv"), index=False,
    )


def store_tetrode_results():
    args = SimpleNamespace(
        max_depth=1,
        num_cpus=1,
        cfg="tetrode_ca3_ca11",
        clt_start=30,
        subsample_rate=0,
        approx_hypergeo=False,
    )
    result = ctrl_main(parse_cfg("tetrode_ca3_ca1.cfg"), args)

    df = df_from_dict(
        result["mpf"]["total"],
        cols=["Number of sampled connected neurons", "Probability"],
    )
    df.to_csv(
        os.path.join(here, "..", "results", "tetrode_man.csv"), index=False,
    )


def store_npix_results():
    args = SimpleNamespace(
        max_depth=1,
        num_cpus=1,
        cfg="ca3_ca1",
        clt_start=10,
        subsample_rate=0.01,
        approx_hypergeo=False,
    )
    result = ctrl_main(parse_cfg("ca3_ca1.cfg"), args)

    df = df_from_dict(
        result["mpf"]["total"],
        cols=["Number of sampled connected neurons", "Probability"],
    )
    df.to_csv(
        os.path.join(here, "..", "results", "npix_man.csv"), index=False,
    )


def store_sub_results():
    args = SimpleNamespace(
        max_depth=1,
        num_cpus=1,
        cfg="ca1_sub_high.cfg",
        clt_start=30,
        subsample_rate=0,
        approx_hypergeo=False,
    )
    cfg = parse_cfg("ca1_sub_high.cfg")
    cfg["default"]["num_samples"] = "[20, 20]"
    result = ctrl_main(cfg, args)
    df = df_from_dict(
        result["mpf"]["total"],
        cols=["Number of sampled connected neurons", "Probability"],
    )
    df.to_csv(
        os.path.join(here, "..", "results", "tetrode_sub_high.csv"), index=False,
    )
    args = SimpleNamespace(
        max_depth=1,
        num_cpus=1,
        cfg="ca1_sub_high.cfg",
        clt_start=30,
        subsample_rate=0,
        approx_hypergeo=False,
    )
    cfg = parse_cfg("ca1_sub_low.cfg")
    cfg["default"]["num_samples"] = "[20, 20]"
    result = ctrl_main(cfg, args)
    df = df_from_dict(
        result["mpf"]["total"],
        cols=["Number of sampled connected neurons", "Probability"],
    )
    df.to_csv(
        os.path.join(here, "..", "results", "tetrode_sub_low.csv"), index=False,
    )


def main():
    store_region_results()
    store_tetrode_results()
    store_npix_results()
    store_sub_results()
