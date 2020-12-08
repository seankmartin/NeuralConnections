"""Main entry: control script for parsing config files and running them."""

import os
import argparse
from configparser import ConfigParser
from pprint import pprint

from skm_pyutils.py_config import print_cfg
from .matrix import main as mouse_main
from .stats_convergence_rate import main as stats_main
from .main import main
from .produce_figures import main as compound_main


def run(own_parsed=None):
    """Command line interface."""

    if own_parsed is not None:
        parsed = own_parsed
    else:
        parser = argparse.ArgumentParser(
            description="Connection code command line interface"
        )
        parser.add_argument(
            "-c", "--cfg", type=str, help="Path to cfg file, or name of file"
        )
        parser.add_argument(
            "-n",
            "--num_cpus",
            type=int,
            default=1,
            help="Number of CPUs to use - default 1",
        )
        parser.add_argument(
            "-d", "--max_depth", type=int, default=1, help="Max DFS depth - default 1"
        )
        parser.add_argument(
            "-mc",
            "--mouse",
            action="store_true",
            help="Whether to analyse the mouse connectome data",
        )
        parser.add_argument(
            "-s",
            "--stats",
            action="store_true",
            help="Whether to analyse the stats convergence rate",
        )
        parser.add_argument(
            "-f",
            "--final",
            action="store_true",
            help="Whether to analyse the final plots",
        )
        parser.add_argument(
            "-l",
            "--clt_start",
            type=int,
            default=30,
            help="Start point for CLT - default 30, recommended 10 for mix of speed and accuracy",
        )
        parser.add_argument(
            "-sr",
            "--subsample_rate",
            type=float,
            default=0.01,
            help="Subsample rate for distributions - default 0.01, set to 1 or 0 to turn off subsampling",
        )
        parser.add_argument(
            "-a",
            "--approx_hypergeo",
            action="store_true",
            help="Whether to analyse the approximate hypergeometric by the binomial",
        )

        parsed = parser.parse_args()

    if (parsed.subsample_rate == 0) or (parsed.subsample_rate == 1):
        parsed.subsample_rate = None

    if parsed.mouse:
        print("Analysing mouse connectome data")
        pprint(mouse_main())
        return

    if parsed.stats:
        print("Analysing stats convergence data")
        pprint(stats_main(1000, 100, 30, 20000))
        return

    if parsed.final:
        pprint(compound_main())
        return

    cfg_path = parsed.cfg
    here = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(here, "..", "configs", cfg_path)

    if not os.path.isfile(cfg_path):
        raise ValueError(
            "Non existent cfg passed, options are {}".format(
                os.listdir(os.path.join(here, "..", "configs"))
            )
        )

    cfg = ConfigParser()
    cfg.read(cfg_path)

    print_cfg(cfg, "Program started with configuration")

    return main(cfg, parsed)


if __name__ == "__main__":
    run()
