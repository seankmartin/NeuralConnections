"""Plotting functions."""
from collections import OrderedDict
import os
from math import isclose

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dictances.bhattacharyya import bhattacharyya
from matplotlib.ticker import MaxNLocator
from scipy.stats import skewnorm, norm, uniform, truncexpon

here = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = "figures"
PALETTE = "dark"
LABELSIZE = 12


def load_df(name):
    """Load a pandas dataframe from csv file at results/name."""
    load_name = os.path.join(here, "..", "results", name)
    df = pd.read_csv(load_name)
    return df


def despine():
    """Despine the current plot with trimming."""
    sns.despine(offset=0, trim=True)


def set_p():
    """Set the seaborn palette."""
    sns.set_palette(PALETTE)
    # sns.set_context(
    #     "paper",
    #     rc={
    #         "axes.titlesize": 18,
    #         "axes.labelsize": 14,
    #         "lines.linewidth": 2,
    #     },
    # )
    sns.set_context(
        "paper",
        font_scale=1.5,
        rc={"lines.linewidth": 3.6},
    )


def set_m():
    sns.set_context(
        "paper",
        font_scale=1.4,
        rc={"lines.linewidth": 1.8},
    )


def plot_visp_visl_shift():
    df = load_df("sub_VISp_VISl_depth_1.csv")
    nshifted = df[df["Shifted"] == "original"]
    shifted = df[df["Shifted"] == "shifted"]

    nshifted = nshifted[nshifted["Number of connected neurons"] <= 25]
    o_name = "original-visp-visl-stats.pdf"
    plot_pmf_accuracy(nshifted, o_name)

    shifted = shifted[shifted["Number of connected neurons"] <= 25]
    o_name = "shifted-visp-visl-stats.pdf"
    plot_pmf_accuracy(shifted, o_name)

    df = load_df("sub_VISp_VISl_stats_all_ds.csv")
    nshifted = df[df["Shifted"] == "original"]
    shifted = df[df["Shifted"] == "shifted"]

    o_name = "original-visp-visl-sub-all-depths.pdf"
    plot_pmf_depth(nshifted, o_name)
    o_name = "shifted-visp-visl-sub-all-depths.pdf"
    plot_pmf_depth(shifted, o_name)


def save(fig, out_name):
    """Save the figure to figures/out_name."""
    out_path = os.path.abspath(os.path.join(here, "..", OUTPUT_DIR, out_name))
    print("Saving figure to {}".format(out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fig is not None:
        fig.savefig(out_path, dpi=400)
    else:
        plt.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_samples_v_prop(df, out_name="depth_plot.pdf"):
    """Plot the number of samples against proportion of connections."""
    fig, ax = plt.subplots()
    set_p()
    df["Max distance"] = df["Max distance"].astype("category")
    sns.lineplot(
        x="Number of samples",
        y="Proportion of connections",
        data=df,
        style="Max distance",
        hue="Max distance",
        ax=ax,
        ci=None,
    )
    ax.set_xlabel("Number of samples", fontsize=LABELSIZE)
    ax.set_ylabel("Expected proportion connected", fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_pmf(df, out_name, full=False):
    """Plot the pmf from the given dataframe."""
    fig, ax = plt.subplots()
    set_m()
    x = df["Number of sampled connected neurons"]
    y = df["Probability"]

    ax.plot(x, y, "ko", ms=2.5)
    y_vals_min = [0 for _ in x]
    y_vals_max = y
    colors = ["k" for _ in x]
    if len(x) < 30:
        ax.set_xticks([i for i in range(len(x))])
        ax.set_xticklabels([i for i in range(len(x))])

    if full:
        ax.set_ylim([0, 1])

    ax.vlines(x, y_vals_min, y_vals_max, colors=colors)
    plt.xlabel("Number of sampled connected neurons", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_pmf_depth(df, out_name, full=False):
    """Plot the pmf from the given dataframe."""
    fig, ax = plt.subplots()
    set_p()
    df["Max distance"] = df["Max distance"].astype("category")
    sns.lineplot(
        x="Number of connected neurons",
        y="Probability",
        hue="Max distance",
        style="Max distance",
        ci=None,
        data=df,
        ax=ax,
    )

    if full:
        ax.set_ylim([0, 1])

    plt.xlabel("Number of sampled connected neurons", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_connection_samples(df, out_name):
    """Plot the connection samples from the dataframe."""
    fig, ax = plt.subplots()
    set_p()
    sns.lineplot(
        x="Number of samples",
        y="Proportion of connections",
        data=df,
        style="Max distance",
        hue="Max distance",
        ax=ax,
    )
    plt.xlabel("Number of samples", fontsize=LABELSIZE)
    plt.ylabel("Proportion of connections", fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_pmf_accuracy(df, out_name):
    """Plot the accuracy of the PMF from the dataframe."""
    fig, ax = plt.subplots()
    set_p()

    calculation_vals = df["Calculation"].values
    has_numbers = False
    greater_than_one = False
    hue = "Calculation"
    for val in calculation_vals:
        if val.endswith("1"):
            has_numbers = True
        if val.endswith("2"):
            greater_than_one = True

    if has_numbers:
        df["Max geodesic distance"] = df.apply(
            lambda row: str(row.Calculation.split(" ")[-1]), axis=1
        )
        df["Calculation"] = df.apply(lambda row: row.Calculation[:-2], axis=1)
        hue = "Max geodesic distance"

    df = df.sort_values(by=["Calculation"], ascending=False)

    if greater_than_one:
        sns.lineplot(
            x="Number of connected neurons",
            y="Probability",
            hue=hue,
            style="Calculation",
            ci=None,
            data=df,
            ax=ax,
        )
    else:
        sns.lineplot(
            x="Number of connected neurons",
            y="Probability",
            hue="Calculation",
            style="Calculation",
            ci=None,
            data=df,
            ax=ax,
        )
    plt.xlabel("Number of connected neurons", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_pmf_comp(dfs, names, out_name):
    """Plot the comparison of pmfs in the dataframe."""
    df = None
    for df_i, name_i in zip(dfs, names):
        df_i["Connectivity"] = [name_i for _ in range(len(df_i))]
        if df is None:
            df = df_i
        else:
            df = df.append(df_i)
    fig, ax = plt.subplots()
    set_p()
    sns.lineplot(
        x="Number of sampled connected neurons",
        y="Probability",
        hue="Connectivity",
        style="Connectivity",
        data=df,
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    plt.xlabel("Number of sampled connected neurons", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_pmf_converge(df, out_name):
    """Plot the comparison of pmfs in the dataframe."""
    fig, ax = plt.subplots()
    set_p()
    sns.lineplot(
        x="Number of sampled connected neurons",
        y="Probability",
        hue="Calculation",
        style="Calculation",
        data=df,
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    plt.xlabel("Number of sampled connected neurons", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_exp_comp(dfs, names, out_name, prop=False, depth=False):
    """Plot the accuracy of the expected value."""
    df = None
    for df_i, name_i in zip(dfs, names):
        df_i["Connectivity"] = [name_i for _ in range(len(df_i))]
        if df is None:
            df = df_i
        else:
            df = df.append(df_i)
    if prop:
        y_name = "Expected proportion connected"
    else:
        y_name = "Expected connected"

    style = "Connectivity" if not depth else "Max distance"
    fig, ax = plt.subplots()
    set_p()
    sns.lineplot(
        x="Number of samples",
        y=y_name,
        data=df,
        style=style,
        hue="Connectivity",
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    plt.xlabel("Number of samples", fontsize=LABELSIZE)
    plt.ylabel(y_name, fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_exp_accuracy(df, out_name, prop=False, split=True):
    """Plot the accuracy of the expected value."""
    if prop:
        y_name = "Expected proportion connected"
    else:
        y_name = "Expected connected"
    if split:
        hue = "Max distance"
    else:
        hue = "Calculation"
    fig, ax = plt.subplots()
    set_p()
    sns.lineplot(
        x="Number of samples",
        y=y_name,
        data=df,
        style="Calculation",
        hue=hue,
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    plt.xlabel("Number of samples", fontsize=LABELSIZE)
    plt.ylabel(y_name, fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_region_vals(df, out_name, x_name="Connectivity", scale=(10, 4)):
    """Plot region specific values from the dataframe."""
    fig, ax = plt.subplots(figsize=scale)
    set_p()

    sns.barplot(
        x=x_name,
        y="Expected proportion connected",
        hue="Calculation",
        data=df,
        ax=ax,
    )
    plt.xlabel(x_name, fontsize=LABELSIZE)
    plt.ylabel("Expected proportion connected", fontsize=LABELSIZE)
    despine()
    save(fig=None, out_name=out_name)


def plot_bhattacharyya(regions, out_name, max_depth=1):
    res = []
    for i, r in enumerate(regions):
        fname = f"sub_regions_{r[0]}_{r[1]}_depth_{max_depth}.csv"
        df = load_df(fname)
        mc = df[df["Calculation"] == "Monte Carlo simulation"]
        mc_dict = {}
        for row in mc.itertuples():
            mc_dict[row._1] = row.Probability

        stats = df[df["Calculation"] == "Statistical estimation"]
        stats_dict = {}
        for row in stats.itertuples():
            stats_dict[row._1] = row.Probability
        distance = bhattacharyya(mc_dict, stats_dict)
        res.append([distance, df["Regions"][0]])

    df = pd.DataFrame(res, columns=["Bhattacharyya distance", "Regions"])
    df.to_csv(os.path.join(here, "..", "results", "bhattacharyya.csv"), index=False)

    fig, ax = plt.subplots()
    set_p()
    sns.barplot(y="Bhattacharyya distance", x="Regions", data=df, ax=ax)
    despine()
    save(fig=None, out_name=out_name)


def plot_region_sim(df, out_name, x_name="Connectivity", scale=(10, 4)):
    """Plot region specific values from the dataframe."""
    fig, ax = plt.subplots(figsize=scale)
    set_p()

    sns.barplot(
        x=x_name,
        y="Bhattacharyya distance",
        data=df,
        ax=ax,
    )
    plt.xlabel(x_name, fontsize=LABELSIZE)
    plt.ylabel("Bhattacharyya distance", fontsize=LABELSIZE)
    despine()
    save(fig=None, out_name=out_name)


def plot_distribution(dist, out_name):
    """Plot the pmf given by the distribution."""
    fig, ax = plt.subplots()
    set_p()
    x = list(dist.keys())
    y = list(dist.values())
    ax.plot(x, y, "ko", ms=2.5)
    y_vals_min = [0 for _ in x]
    y_vals_max = y
    colors = ["k" for _ in x]
    ax.vlines(x, y_vals_min, y_vals_max, colors=colors)

    plt.xlabel("Value", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)

    despine()
    save(fig, out_name)


def plot_large_dist(df, out_name):
    """Plot pmf for a large distribution"""
    fig, ax = plt.subplots()
    set_p()

    sns.lineplot(data=df, x="Outgoing connections", y="Probability", ax=ax)
    despine()
    save(fig, out_name)


def plot_power_law(df, out_name):
    """Plot power law"""
    fig, ax = plt.subplots()
    set_p()

    sns.scatterplot(
        data=df,
        x="Size of each brain region",
        y="Number of required samples",
        style="Connections",
        hue="Connections",
    )
    despine()
    save(fig, out_name)


def plot_acc_interp(x_samps, interped_vals, xvals, yvals, out_name, true_y=None):
    """Plot the accuracy of interpolation."""
    fig, ax = plt.subplots()
    set_p()
    ax.plot(x_samps, interped_vals, c="b", linestyle="-", label="interp")
    ax.plot(xvals, yvals, "gx", label="samples", markersize="3.0")
    if true_y is not None:
        ax.plot(x_samps, true_y, c="r", linestyle="--", label="true")
    plt.legend()
    plt.xlabel("Number of receivers in B", fontsize=LABELSIZE)
    plt.ylabel("Weighted probability", fontsize=LABELSIZE)
    save(fig, out_name)


def discretised_rv(rv, min_, max_, middle=False):
    """Discretise a continous RV into min max range"""
    right_shift = 0.5 if middle else 1
    left_shift = 0.5 if middle else 0
    od = OrderedDict()
    for val in range(min_, max_ + 1):
        od[val] = rv.cdf(min(max_, val + right_shift)) - rv.cdf(
            max(min_, val - left_shift)
        )
    sum_ = sum(od.values())
    if not isclose(sum_, 1.0):
        for k, v in od.items():
            od[k] = v / sum_

    sum_ = sum(od.values())
    if not isclose(sum_, 1.0):
        raise ValueError(f"Distribution discrete does not sum to 1.0, got {sum_}")
    return od


def plot_dist_explain(dfs, out_names):
    """Plot the explanation of computing the distributions."""
    fig, ax = plt.subplots()
    set_p()
    df = dfs[0]
    sns.lineplot(x="Number of sampled senders", y="Probability", ax=ax, data=df)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    plt.xlabel("Number of sampled senders", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_names[0])

    fig, ax = plt.subplots(figsize=(7, 5.2))
    set_p()
    df = dfs[1]
    sns.lineplot(x="Number of receivers", y="Probability", ax=ax, data=df)
    plt.xlabel("Number of receivers", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_names[1])

    fig, ax = plt.subplots(figsize=(7, 5.2))
    set_p()
    df = dfs[2]
    st = [0, 4, 8, 12, 16, 20]
    conds = []
    for val in st:
        conds.append(df["Number of sampled A"] == val)
    final_filt = conds[0]
    for val in conds[1:]:
        final_filt = final_filt | val

    sns.lineplot(
        x="Number of receivers",
        y="Probability",
        ax=ax,
        data=df[final_filt].astype({"Number of sampled A": "str"}),
        style="Number of sampled A",
        hue="Number of sampled A",
    )
    plt.xlabel("Number of receivers", fontsize=LABELSIZE)
    plt.ylabel("Weighted probability", fontsize=LABELSIZE)
    despine()
    save(fig, out_names[2])

    fig, ax = plt.subplots()
    set_p()
    df = dfs[3]
    sns.lineplot(x="Number of sampled receivers", y="Probability", ax=ax, data=df)
    plt.xlabel("Number of sampled receivers", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    despine()
    save(fig, out_names[3])


def plot_dist_accuracy(rv, df, out_name):
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(list(rv.keys()), list(rv.values()))
    # ax[0].set_xlabel("Forward connections")
    # ax[0].set_ylabel("Probability")
    # despine()
    fig, ax = plt.subplots()

    sns.lineplot(
        data=df,
        ax=ax,
        x="Number of recorded connected neurons",
        y="Probability",
        hue="Calculation",
        style="Calculation",
    )
    despine()

    save(fig, out_name)


def main():
    """Defines the plots performed in produce_figures - without performing analysis."""
    print("Starting main plotting")

    # Figure 1
    plot_pmf(load_df("explain_fig2.csv"), "explain_fig_pmf.pdf")

    # Figure 2
    plot_visp_visl_shift()

    # Figure 3
    df = load_df("mouse_region_exp_probes.csv")
    plot_region_vals(
        df,
        "mouse_region_exp_probes.pdf",
        x_name="Regions",
        scale=(12, 5),
    )
    df = df[df["Calculation"] == "Monte Carlo simulation"]
    plot_region_vals(
        df,
        "mouse_region_exp_probes_graph.pdf",
        x_name="Regions",
        scale=(12, 5),
    )

    # Figure 4
    plot_samples_v_prop(load_df("samples_depth_ca3_ca1.csv"), "ca3_ca1_samps_depth.pdf")
    plot_pmf(load_df("npix_probe_ca3_ca1.csv"), "npix_pmf.pdf")
    df_list = [
        load_df("connection_samples_hc_high.csv"),
        load_df("Connection_samples_hc_med.csv"),
        load_df("connection_samples_hc_low.csv"),
    ]
    df_names = [
        "1.8% (36% to 5%)",
        "0.5% (10% to 5%)",
        "0.2% (16% to 1.1%)",
    ]
    plot_exp_comp(
        df_list,
        df_names,
        "samples_hc_both.pdf",
        prop=True,
        depth=False,
    )
    dfs = [
        load_df("20_sub_high.csv"),
        load_df("20_sub_out.csv"),
        load_df("20_sub_low.csv"),
    ]
    plot_pmf_comp(dfs, df_names, "ca1_sub_tet_comp.pdf")

    ## Extra figures

    # Mouse plots for full regions
    plot_samples_v_prop(load_df("MOp_to_SSP-ll_depth.csv"), "mouse_samps.pdf")
    plot_region_vals(
        load_df("mouse_region_exp_fig.csv"),
        "mouse_region_exp.pdf",
        x_name="Regions",
        scale=(12, 5),
    )

    # Accuracy plots
    plot_exp_accuracy(
        load_df("connection_samples_fig.csv"), "samples_acc.pdf", prop=True
    )
    plot_pmf_accuracy(load_df("pmf_comp_fig.csv"), "d3_acc.pdf")
    plot_pmf_accuracy(load_df("pmf_comp_pmf.csv"), "pmf_acc.pdf")
    plot_region_vals(load_df("exp_man.csv"), "region_acc_man.pdf", scale=(12, 5))
    plot_pmf_accuracy(load_df("MOp_to_SSP-ll_pmf_final_1_79.csv"), "pmf_mouse_acc.pdf")
    plot_exp_accuracy(
        load_df("total_b_exp_fig.csv"), "exp_total_b.pdf", prop=True, split=False
    )

    # Explanation figures - mostly done in other function
    plot_dist_explain(
        [
            load_df("a_prob_eg.csv"),
            load_df("b_prob_eg.csv"),
            load_df("b_each_eg.csv"),
            load_df("b_fin_eg.csv"),
        ],
        [
            "a_prob_eg.pdf",
            "b_prob_eg.pdf",
            "b_each_eg.pdf",
            "b_fin_eg.pdf",
        ],
    )

    # Convergence rate
    plot_pmf_converge(load_df("convergence_pmf.csv"), "stats_converge.pdf")
    plot_pmf_converge(
        load_df("stats_convergence_fixed.csv"), "stats_fixed_converge.pdf"
    )

    # Bhattacharyya
    regions = [
        ("MOp", "SSp-ll"),
        ("SSp-ll", "MOp"),
        ("VISp", "VISl"),
        ("VISl", "VISp"),
        ("AUDp", "AUDpo"),
        ("AUDpo", "AUDp"),
        ("ILA", "PL"),
        ("PL", "ILA"),
    ]
    plot_bhattacharyya(regions, "bhattacharyya_mouse.pdf")

    print(load_df("sub_regions_MOp_SSp-ll_all_depth.csv"))
    plot_samples_v_prop(
        load_df("sub_regions_MOp_SSp-ll_all_depth.csv"), "MOp-SSp-depth.pdf"
    )
    plot_pmf(load_df("tetrode_full.csv"), "ca3_ca1_tetrode_pmf.pdf")

    max_ = 10000
    exp_dist = discretised_rv(truncexpon(10000, scale=500, loc=0), 0, 6000)
    uniform_dist = discretised_rv(uniform(scale=1000, loc=0), 0, 1000)
    norm_dist = discretised_rv(norm(loc=400, scale=400), 0, 2000)
    skewnorm_dist = discretised_rv(skewnorm(loc=0, scale=600, a=40), 0, 2000)
    dists = [exp_dist, uniform_dist, norm_dist, skewnorm_dist]
    names = ["exp", "unif", "norm", "skewnorm"]

    for rv, name in zip(dists, names):
        in_name = f"{name}_accuracy.csv"
        df = load_df(in_name)
        out_name = f"{name}_accuracy_dist.pdf"
        plot_dist_accuracy(rv, df, out_name)

    df1 = load_df("matrix_dist.csv")
    df2 = load_df("power_law.csv")
    name1 = "matrix_dist_plot.pdf"
    name2 = "power_law_plot.pdf"

    plot_large_dist(df1, name1)
    plot_power_law(df2, name2)


if __name__ == "__main__":
    main()
