import argparse

import cProfile, pstats, io
from pstats import SortKey
from neuroconnect.control import run


def main():
    pr = cProfile.Profile()
    pr.enable()
    parser = argparse.ArgumentParser(
        description="Connection code command line interface"
    )
    parser.add_argument(
        "-c", "--cfg", type=str, help="Path to cfg file, or name of file"
    )
    parser.add_argument(
        "-n", "--num_cpus", type=int, default=1, help="Number of CPUs to use"
    )
    parser.add_argument("-d", "--max_depth", type=int, default=1, help="Max DFS depth")
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
        "-f", "--final", action="store_true", help="Whether to analyse the final plots",
    )
    parser.add_argument(
        "-l", "--clt_start", type=int, default=30, help="Start point for CLT"
    )
    parser.add_argument(
        "-sr",
        "--subsample_rate",
        type=float,
        default=0.01,
        help="Subsample rate for distributions",
    )
    parser.add_argument(
        "-a",
        "--approx_hypergeo",
        action="store_true",
        help="Whether to analyse the approximate hypergeometric by the binomial",
    )

    parsed = parser.parse_args()
    run(own_parsed=parsed)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    main()
