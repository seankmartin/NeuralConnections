"""Plotting functions."""
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

here = os.path.dirname(os.path.realpath(__file__))
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
        "paper", font_scale=1.4, rc={"lines.linewidth": 3.2},
    )


def set_m():
    sns.set_context(
        "paper", font_scale=1.4, rc={"lines.linewidth": 1.5},
    )


def save(fig, out_name):
    """Save the figure to figures/out_name."""
    out_path = os.path.abspath(os.path.join(here, "..", "figures", out_name))
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
    sns.lineplot(
        x="Number of samples",
        y="Proportion of connections",
        data=df,
        style="Max distance",
        hue="Max distance",
        ax=ax,
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


def plot_pmf_comp(df1, df2, out_name):
    """Plot the rate of convergence of the dataframe."""
    df1["Connectivity"] = ["High - 90% to 2%" for i in range(len(df1))]
    df2["Connectivity"] = ["Low - 70% to 1%" for i in range(len(df2))]
    df = df1.append(df2)
    fig, ax = plt.subplots()
    set_p()
    sns.lineplot(
        x="Number of sampled connected neurons",
        y="Probability",
        hue="Connectivity",
        style="Connectivity",
        data=df,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    plt.xlabel("Number of sampled connected neurons", fontsize=LABELSIZE)
    plt.ylabel("Probability", fontsize=LABELSIZE)
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
        x="Number of samples", y=y_name, data=df, style="Calculation", hue=hue, ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True, min_n_ticks=10))
    plt.xlabel("Number of samples", fontsize=LABELSIZE)
    plt.ylabel(y_name, fontsize=LABELSIZE)
    despine()
    save(fig, out_name)


def plot_exp_comp(df1, df2, out_name, prop=False):
    """Plot the accuracy of the expected value."""
    df1["Connectivity"] = ["High - 90% to 2%" for i in range(len(df1))]
    df2["Connectivity"] = ["Low - 70% to 1%" for i in range(len(df2))]
    df = df1.append(df2)
    if prop:
        y_name = "Expected proportion connected"
    else:
        y_name = "Expected connected"
    fig, ax = plt.subplots()
    set_p()
    sns.lineplot(
        x="Number of samples",
        y=y_name,
        data=df,
        style="Connectivity",
        hue="Connectivity",
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
        x=x_name, y="Expected proportion connected", hue="Calculation", data=df, ax=ax,
    )
    plt.xlabel(x_name, fontsize=LABELSIZE)
    plt.ylabel("Expected proportion connected", fontsize=LABELSIZE)
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


def main():
    """Defines the plots performed in produce_figures - without performing analysis."""
    print("Starting main plotting")

    # Mouse plots
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

    # Example plots - HC/SUB
    plot_pmf(load_df("tetrode_man.csv"), "tetrode_pmf.pdf")
    plot_pmf(load_df("npix_man.csv"), "npix_pmf.pdf")
    plot_exp_comp(
        load_df("connection_samples_hc_high.csv"),
        load_df("connection_samples_hc_low.csv"),
        "samples_hc_both.pdf",
        prop=True,
    )
    plot_pmf_comp(
        load_df("tetrode_sub_high.csv"),
        load_df("tetrode_sub_low.csv"),
        "ca1_sub_tet_comp.pdf",
    )

    # Explanation figures - mostly done in other function
    plot_dist_explain(
        [
            load_df("a_prob_eg.csv"),
            load_df("b_prob_eg.csv"),
            load_df("b_each_eg.csv"),
            load_df("b_fin_eg.csv"),
        ],
        ["a_prob_eg.pdf", "b_prob_eg.pdf", "b_each_eg.pdf", "b_fin_eg.pdf",],
    )


if __name__ == "__main__":
    main()
