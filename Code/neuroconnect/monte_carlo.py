"""General monte carlo simulation helper."""

import os
from time import time
import multiprocessing
from collections import OrderedDict

from pathos.multiprocessing import ProcessPool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpmath import mpf

global_fn_multi = None
global_var_multi = None


def multiprocessing_func(fn_to_eval, random_var_gen, i):
    """Allows monte carlo to run on multiple CPUs."""
    random_vars = random_var_gen(i)
    result = fn_to_eval(*random_vars)
    return result


def monte_carlo(
    fn_to_eval,
    random_var_gen,
    num_simulations,
    num_cpus=1,
    save_every=1,
    save_name="monte_carlo_result.csv",
    headers=None,
    progress=True,
):
    """
    Full monte carlo simulation loop.

    Evaluate fn_to_eval over num_simulations iterations, with
    *random_var_gen(i) passed into fn_to_eval at each iteration i.

    """
    all_stats = []
    global_fn_multi = fn_to_eval
    global_var_multi = random_var_gen
    save_every = int(save_every * num_simulations)

    # temp = [random_var_gen() for _ in range(num_simulations)]
    # random_vars = [[] for _ in temp[0]]
    # for val in temp:
    #     for i, item in enumerate(val):
    #         random_vars[i].append(item)

    pbar = tqdm(range(num_simulations), disable=not progress)
    if num_cpus > 1:
        # pool = multiprocessing.get_context("spawn").Pool(num_cpus)
        pool = ProcessPool(nodes=num_cpus)
        print(
            "Launching {} workers for {} iterations".format(num_cpus, num_simulations)
        )
        pbar.set_description("Monte carlo loop")
        for i in pbar:
            result = pool.apipe(
                multiprocessing_func, global_fn_multi, global_var_multi, i
            )
            # result = pool.amap(fn_to_eval, random_vars)
            # result = pool.apply_async(
            #     multiprocessing_func,
            #     (i, global_fn_multi, global_var_multi))
            all_stats.append(result.get())

    else:
        pbar.set_description("Monte carlo loop")
        for i in pbar:
            random_vars = random_var_gen(i)
            result = fn_to_eval(*random_vars)
            all_stats.append(result)

            if (i != 0) and (i % save_every == 0):
                parts = os.path.splitext(save_name)
                out_name = parts[0] + "_" + str(i) + parts[1]
                df = list_to_df(all_stats, headers)
                here = os.path.dirname(os.path.realpath(__file__))
                os.makedirs(os.path.join(here, "..", "mc"), exist_ok=True)
                print("Saving results at {} iterations to {}".format(i, out_name))
                df.to_csv(os.path.join(here, "..", "mc", out_name), index=False)

    return all_stats


def list_to_df(in_list, headers=None):
    """Convert a list to a dataframe with the given headers."""
    if headers is None:
        headers = ["V{}".format(i) for i in range(len(in_list[0]))]
    results_df = pd.DataFrame.from_records(in_list, columns=headers)
    return results_df


def summarise_monte_carlo(
    df, txt_outfile=None, plot=True, to_plot=None, plt_outfile=None, do_print=False,
):
    """Summary stats of monte carlo with optional dist plot."""
    result = df.describe().round(4)
    if (txt_outfile is None) and do_print:
        print(result)
    elif txt_outfile is not None:
        with open(txt_outfile, "w") as f:
            f.write(result)
    if plot:
        if to_plot is None:
            raise ValueError("Please provide a column to plot")
        a = df[to_plot].to_numpy()
        is_unique = (a[0] == a).all()
        if not is_unique:
            sns.displot(
                df[to_plot],
                kde=True,
                rug=False,
                # kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                # hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"},
            )
            if plt_outfile is None:
                plt.show()
            else:
                plt.savefig(plt_outfile, dpi=400)
        plt.close()
    return result


def get_distribution(result_df, column_name, num_iters):
    """Calculate the simulated distribution of column_name."""
    distrib = {}
    to_add = 1 / num_iters
    for val in result_df[column_name]:
        if val in distrib:
            distrib[val] += to_add
        else:
            distrib[val] = to_add

    ordered_dist = OrderedDict()
    keys = sorted(distrib.keys())
    for key in keys:
        ordered_dist[key] = distrib[key]

    return ordered_dist


def dist_difference(actual_distribution, expected_distribution):
    """Calculate the difference between two distributions."""
    difference = {}
    for k, v in expected_distribution.items():
        difference[k] = actual_distribution[k] - v
    return difference
