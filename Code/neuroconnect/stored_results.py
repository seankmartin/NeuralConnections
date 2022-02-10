"""A module to store some results that are parsed from .txt files."""

import os
from configparser import ConfigParser
from types import SimpleNamespace

import pandas as pd
import numpy as np
from skm_pyutils.py_table import list_to_df

from dictances.bhattacharyya import bhattacharyya

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
    np.random.seed(42)
    vals = []
    names = [
        "Tetrode CA3 CA1",
        "MOp to SSp-ll",
        "Figure 1 E",
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
        os.path.join(here, "..", "results", "exp_man.csv"),
        index=False,
    )


def store_tetrode_results():
    np.random.seed(42)
    args = SimpleNamespace(
        max_depth=1,
        num_cpus=1,
        cfg="tetrode_ca3_ca1",
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
        os.path.join(here, "..", "results", "tetrode_man.csv"),
        index=False,
    )


def store_tetrode_results_full():
    np.random.seed(42)
    args = SimpleNamespace(
        max_depth=1,
        num_cpus=1,
        cfg="tetrode_ca3_ca1_full",
        clt_start=30,
        subsample_rate=0,
        approx_hypergeo=False,
    )
    result = ctrl_main(parse_cfg("tetrode_ca3_ca1_full.cfg"), args)

    df = df_from_dict(
        result["mpf"]["total"],
        cols=["Number of sampled connected neurons", "Probability"],
    )
    df.to_csv(
        os.path.join(here, "..", "results", "tetrode_full.csv"),
        index=False,
    )


def store_tetrode_results_depth():
    np.random.seed(42)
    num_samples_range = np.arange(60)
    res_list = []
    headers = ["Number of samples", "Proportion of connections", "Max distance"]
    for depth in (1, 2, 3):
        for s in num_samples_range:
            args = SimpleNamespace(
                max_depth=depth,
                num_cpus=1,
                cfg="tetrode_ca3_ca1_full_stats",
                clt_start=30,
                subsample_rate=0,
                approx_hypergeo=False,
            )
            cfg = parse_cfg("tetrode_ca3_ca1_full.cfg")
            cfg["default"]["num_samples"] = f"[{s}, {s}]"
            result = ctrl_main(cfg, args)
            exp = result["mpf"]["expected"] / s
            res_list.append([s, exp, depth])

    df = list_to_df(res_list, headers=headers)
    df.to_csv(
        os.path.join(here, "..", "results", "samples_depth_ca3_ca1.csv"),
        index=False,
    )


def store_npix_results():
    np.random.seed(42)
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
        os.path.join(here, "..", "results", "npix_man.csv"),
        index=False,
    )


def store_sub_results():
    np.random.seed(42)
    configs = [
        "ca1_sub_high.cfg",
        "ca1_sub_high_out.cfg",
        "ca1_sub_low.cfg",
        "ca1_sub_vhigh.cfg",
    ]
    out_names = [
        "20_sub_high.csv",
        "20_sub_out.csv",
        "20_sub_low.csv",
        "20_sub_vhigh.csv",
    ]

    for cfg_name, name in zip(configs, out_names):
        args = SimpleNamespace(
            max_depth=1,
            num_cpus=1,
            cfg=cfg_name,
            clt_start=30,
            subsample_rate=0.01,
            approx_hypergeo=False,
        )
        cfg = parse_cfg(cfg_name)
        cfg["default"]["num_samples"] = "[20, 20]"
        result = ctrl_main(cfg, args)
        df = df_from_dict(
            result["mpf"]["total"],
            cols=["Number of sampled connected neurons", "Probability"],
        )
        df.to_csv(
            os.path.join(here, "..", "results", name),
            index=False,
        )


def store_mouse_result():
    dict_a = {
        0: 0.097727929,
        1: 0.073771117,
        2: 0.09002461,
        3: 0.098312369,
        4: 0.097549236,
        5: 0.090365064,
        6: 0.079501495,
        7: 0.067243357,
        8: 0.055259551,
        9: 0.044579481,
        10: 0.035678545,
        11: 0.02861985,
        12: 0.02320665,
        13: 0.019120951,
        14: 0.016025042,
        15: 0.013623944,
        16: 0.01169026,
        17: 0.010066428,
        18: 0.00865235,
        19: 0.007391084,
        20: 0.006253293,
        21: 0.005227931,
        22: 0.004312818,
        23: 0.003508591,
        24: 0.002815313,
        25: 0.002230014,
        26: 0.001745977,
        27: 0.001353389,
        28: 0.001040298,
        29: 0.000794062,
        30: 0.000602467,
        31: 0.000454566,
        32: 0.000341045,
        33: 0.0002543,
        34: 0.000188282,
        35: 0.000138257,
        36: 0.000100549,
        37: 7.23e-05,
        38: 5.13e-05,
        39: 3.59e-05,
        40: 2.47e-05,
        41: 1.66e-05,
        42: 1.10e-05,
        43: 7.07e-06,
        44: 4.43e-06,
        45: 2.69e-06,
    }
    dict_b = {
        0: 0.0755,
        1: 0.07796,
        2: 0.09164,
        3: 0.09948,
        4: 0.09346,
        5: 0.08536,
        6: 0.07598,
        7: 0.06552,
        8: 0.05336,
        9: 0.04534,
        10: 0.03604,
        11: 0.0309,
        12: 0.02306,
        13: 0.02202,
        14: 0.0174,
        15: 0.01518,
        16: 0.0133,
        17: 0.01316,
        18: 0.01104,
        19: 0.0089,
        20: 0.00808,
        21: 0.00752,
        22: 0.00576,
        23: 0.00448,
        24: 0.00386,
        25: 0.00368,
        26: 0.00252,
        27: 0.0019,
        28: 0.00158,
        29: 0.00144,
        30: 0.0009,
        31: 0.001,
        32: 0.00066,
        33: 0.00048,
        34: 0.00042,
        35: 0.00016,
        36: 0.00022,
        37: 0.00022,
        38: 0.00018,
        39: 0.00012,
        40: 6.00e-05,
        41: 4.00e-05,
        42: 4.00e-05,
        43: 6.00e-05,
        45: 2.00e-05,
    }

    dist = bhattacharyya(dict_a, dict_b)
    return dist


def main():
    store_region_results()
    store_tetrode_results()
    store_npix_results()
    store_sub_results()
    store_mouse_result()
