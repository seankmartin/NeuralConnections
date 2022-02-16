"""
Produce all the paper figures.

See also plot.py
"""
import os
from configparser import ConfigParser
from types import SimpleNamespace

import typer
import pandas as pd
import myterial

from .compound import (
    connections_dependent_on_samples,
    proportion,
    pmf_accuracy,
    connections_dependent_on_regions,
    distance_dependent_on_regions,
    mouse_region_exp,
    out_exp,
    explain_calc,
    mouse_region_exp_probes,
)
from .matrix import main as mouse_main
from .main import main as ctrl_main
from .plot import (
    plot_exp_accuracy,
    load_df,
    plot_pmf_accuracy,
    plot_samples_v_prop,
    plot_region_vals,
    plot_region_sim,
    plot_dist_explain,
    plot_pmf,
    plot_exp_comp,
    plot_pmf_comp,
    plot_visp_visl_shift,
)
from .stats_convergence_rate import (
    test_network_convergence,
    test_config_convergence,
    test_rand_network_convergence,
)
from .stored_results import (
    store_region_results,
    store_tetrode_results,
    store_tetrode_results_full,
    store_tetrode_results_depth,
    store_npix_results,
    store_sub_results,
)
from .atlas_graph import plot_subset_vis

here = os.path.dirname(os.path.realpath(__file__))
app = typer.Typer()


def parse_cfg(name):
    """Parse the configs at configs/name."""
    cfg_path = os.path.join(here, "..", "configs", name)
    cfg = ConfigParser()
    cfg.read(cfg_path)

    return cfg


def mo_to_ss_vis():
    """Visualise MoP to SSp."""
    print("Plotting MOp to SSp-ll connectivity")
    mouse_main(
        A_name="MOp",
        B_name="SSp-ll",
        vis_connect=True,
        final=False,
        analyse=False,
        subsample=False,
    )


def vis_to_vis_vis():
    """Visualise VISp to VISl."""
    print("Plotting VISp to VISl connectivity")
    mouse_main(
        A_name="VISp",
        B_name="VISl",
        vis_connect=True,
        final=False,
        analyse=False,
        subsample=False,
    )


def mo_to_ss_acc():
    """Accuracy of MOp to SSp."""
    mouse_main(
        A_name="MOp",
        B_name="SSp-ll",
        vis_connect=False,
        final=True,
        analyse=False,
        subsample=False,
        num_sampled=[79, 79],
        max_depth=1,
        num_iters=0,
        do_graph=False,
    )


@app.command()
def do_mouse(do_mat_vis=True, do_comp=True, do_exp=True):
    """Produce figures related to blue brain data."""
    if do_mat_vis:
        mo_to_ss_vis()
        vis_to_vis_vis()
    if do_comp:
        mo_to_ss_acc()
        plot_samples_v_prop(load_df("MOp_to_SSP-ll_depth.csv"), "mouse_samps.pdf")
        # Not using pmf currently
    if do_exp:
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
        depths = [1, 1, 1, 1, 1, 1, 1, 1]
        num_samples = [79, 79]
        mouse_region_exp(
            regions, depths, "fig", num_samples, num_iters=1, do_graph=False
        )
        plot_region_vals(
            load_df("mouse_region_exp_fig.csv"),
            "mouse_region_exp.pdf",
            x_name="Regions",
        )


@app.command()
def do_accuracy(
    do_samples: bool = True,
    do_pmf: bool = True,
    do_regions: bool = False,
    do_mouse_acc: bool = False,
    do_exp: bool = True,
    do_growth: bool = False,
):
    """Produce figures related to method accuracy."""
    print("Plotting figures related to accuracy")

    if do_samples:
        connections_dependent_on_samples(
            parse_cfg("recurrent_fig1.cfg"),
            "fig",
            num_iters=1000,
            use_mean=False,
            num_graphs=10,
            sr=0.01,
            clt_start=30,
            fin_depth=3,
        )
        plot_exp_accuracy(
            load_df("connection_samples_fig.csv"), "samples_acc.pdf", prop=True
        )

    if do_pmf:
        pmf_accuracy(
            parse_cfg("d3.cfg"),
            "fig",
            clt_start=30,
            num_iters=1000,
            num_graphs=100,
            do_the_stats=False,
            sr=None,
        )
        plot_pmf_accuracy(load_df("pmf_comp_fig.csv"), "d3_acc.pdf")

        pmf_accuracy(
            parse_cfg("pmf_acc.cfg"),
            "pmf",
            clt_start=30,
            num_iters=1000,
            num_graphs=100,
            depth_full=1,
            sr=None,
        )
        plot_pmf_accuracy(load_df("pmf_comp_pmf.csv"), "pmf_acc.pdf")

    if do_regions:
        # This needs to be updated with correct configs, use stored for now
        # cfg_names = [
        #     "sub_hc.cfg",
        #     "sub_hc.cfg",
        #     "recurrent_fig.cfg",
        #     "recurrent_fig1.cfg",
        # ]
        # r_names = ["Sub HC direct", "Sub HC two", "Highly connected", "Dispersed"]
        # depths = [1, 2, 1, 1]
        # connections_dependent_on_regions(
        #     cfg_names, r_names, depths, "fig", num_iters=20000,
        # )
        # plot_region_vals(load_df("region_exp_fig.csv"), "region_acc.pdf")

        cfg_names = [
            "tetrode_ca3_ca1.cfg",
            "USE STORED MOUSE",
            "recurrent_fig1.cfg",
            "d3.cfg",
            "v_small.cfg",
        ]
        r_names = [
            "Tetrode CA3 CA1",
            "MOp to SSp-ll",
            "Figure 1 E",
            "Max distance 3",
            "Figure 1 A",
        ]
        depths = [1, 1, 1, 3, 1]
        distance_dependent_on_regions(
            cfg_names,
            r_names,
            depths,
            "fig",
            num_iters=20000,
        )
        plot_region_sim(load_df("region_bhatt_fig.csv"), "region_acc_bhatt.pdf")

        print("WARNING: this uses stored results, check correct")
        store_region_results()
        plot_region_vals(load_df("exp_man.csv"), "region_acc_man.pdf")

    if do_mouse_acc:
        mouse_main(
            A_name="MOp",
            B_name="SSp-ll",
            vis_connect=False,
            final=False,
            analyse=True,
            subsample=False,
            num_sampled=[79, 79],
            max_depth=1,
            num_iters=50000,
            do_graph=True,
        )
        plot_pmf_accuracy(
            load_df("MOp_to_SSP-ll_pmf_final_1_79.csv"), "pmf_mouse_acc.pdf"
        )

    if do_exp:
        out_exp(parse_cfg("recurrent_fig1.cfg"), "fig", 1, num_iters=10000)
        plot_exp_accuracy(
            load_df("total_b_exp_fig.csv"), "exp_total_b.pdf", prop=True, split=False
        )

    if do_growth:
        print("Not yet implemented growth plot")


@app.command()
def do_examples(do_exp=True, do_pmf=True, do_types=True):
    """Produce figures related to examples."""
    if do_exp:
        print("Not yet implemented expected number of connected")
    if do_pmf:
        store_tetrode_results()
        plot_pmf(load_df("tetrode_man.csv"), "tetrode_pmf.pdf")
        store_npix_results()
        plot_pmf(load_df("npix_man.csv"), "npix_pmf.pdf")
    if do_types:
        kwargs = {
            "num_iters": 1000,
            "do_graph": False,
            "use_mean": False,
            "num_graphs": 0,
            "sr": 0.01,
            "clt_start": 10,
            "fin_depth": 1,
        }
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_high.cfg"), "hc_high", **kwargs
        )
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_high_out.cfg"), "hc_high_out", **kwargs
        )
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_low.cfg"), "hc_low", **kwargs
        )
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_vhigh.cfg"), "hc_vhigh", **kwargs
        )
        df_list = [
            load_df("connection_samples_hc_vhigh.csv"),
            load_df("connection_samples_hc_high.csv"),
            load_df("Connection_samples_hc_high_out.csv"),
            load_df("connection_samples_hc_low.csv"),
        ]
        df_names = [
            "2.8% (90% to 3.1%)",
            "1.8% (90% to 2%)",
            "1.8% (36% to 5%)",
            "0.8% (70% to 1%)",
        ]
        plot_exp_comp(
            df_list,
            df_names,
            "samples_hc_both.pdf",
            prop=True,
        )
        store_sub_results()
        dfs = [
            load_df("20_sub_vhigh.csv"),
            load_df("20_sub_high.csv"),
            load_df("20_sub_out.csv"),
            load_df("20_sub_low.csv"),
        ]
        plot_pmf_comp(dfs, df_names, "ca1_sub_tet_comp.pdf")


@app.command()
def do_explain(do_vis=True, do_pmf=True, do_dist=True):
    """Produce figures to explain the problem at hand."""
    if do_vis:
        print("Plotting figures for explaining the problem")
        args = SimpleNamespace(
            max_depth=1,
            num_cpus=1,
            cfg="explain_fig1",
            clt_start=2,
            subsample_rate=0.01,
            approx_hypergeo=False,
        )
        ctrl_main(parse_cfg("recurrent_fig.cfg"), args)

        args = SimpleNamespace(
            max_depth=1,
            num_cpus=1,
            cfg="explain_fig2",
            clt_start=2,
            subsample_rate=0.01,
            approx_hypergeo=False,
        )
        res = ctrl_main(parse_cfg("recurrent_fig1.cfg"), args)
        dist = res["mpf"]["total"]
        vals = []
        for k, v in dist.items():
            vals.append([k, v])

        cols = ["Number of sampled connected neurons", "Probability"]
        df = pd.DataFrame(vals, columns=cols)
        output_location = os.path.join(here, "..", "results", "explain_fig2.csv")
        df.to_csv(output_location, index=False)

        df = pd.read_csv(output_location)
        plot_pmf(df, "explain_fig_pmf.pdf")

    if do_pmf:
        proportion(
            parse_cfg("recurrent_fig1.cfg"),
            depths=[1],
        )

    if do_dist:
        explain_calc(parse_cfg("recurrent_fig1.cfg"), out_name="eg", sr=0.01)
        dfs = [
            load_df("a_prob_eg.csv"),
            load_df("b_prob_eg.csv"),
            load_df("b_each_eg.csv"),
            load_df("b_fin_eg.csv"),
        ]
        names = [
            "a_prob_eg.pdf",
            "b_prob_eg.pdf",
            "b_each_eg.pdf",
            "b_fin_eg.pdf",
        ]
        plot_dist_explain(dfs, names)


@app.command()
def do_sub(do_full_vis: bool = False, do_probability: bool = True):
    """Subset figures with probes."""
    names = [
        "full_matrix_vis_VISl_VISp.pdf",
        "sub_matrix_vis_VISl_VISp.pdf",
        "probe_matrix_vis_visl_VISp.pdf",
        "brainrender_visl_visp",
    ]
    colors = [
        myterial.blue_dark,
        myterial.pink_darker,
        myterial.indigo_dark,
        myterial.salmon_darker,
    ]
    # This is for right hemi
    # region_sizes = [391292, 55816]
    # This is for left hemi
    region_sizes = [333055, 49569]
    num_sampled = [79, 79]
    block_size = 10
    simulation_kwargs = dict(max_depth=1, num_cpus=1, num_iters=50000)
    plot_subset_vis(
        names,
        ["VISp", "VISl"],
        region_sizes,
        num_sampled,
        colors=colors,
        do_full_vis=do_full_vis,
        do_probability=do_probability,
        block_size_sub=block_size,
        hemisphere="left",
        **simulation_kwargs,
    )
    plot_visp_visl_shift()


@app.command()
def do_mouse_regions(vis_only: bool = True):
    """Mouse expected with probes through COM and 79 cells"""
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
    # Rotation order is zyx
    # This means the first angle moves left (+), right (-)
    # The second angle moves left (+), right (-)
    # The third angle moves up (+), down (-)
    probe_kwargs = [
        dict(top_scale=0.95, angles_top=[0, 0, 3], angles_bottom=[0, 0, -2]),
        dict(top_scale=0.95, angles_top=[0, 0, 3], angles_bottom=[0, 0, -2]),
        dict(top_scale=0.35, angles_top=[0, 0, 10], angles_bottom=[0, 0, -5]),
        dict(top_scale=0.35, angles_top=[0, 0, 10], angles_bottom=[0, 0, -5]),
        dict(top_scale=0.3, angles_top=[0, 0, 0], angles_bottom=[0, 0, 0]),
        dict(top_scale=0.3, angles_top=[0, 0, 0], angles_bottom=[0, 0, 0]),
        dict(top_scale=0.5, angles_top=[0, 0, 10], angles_bottom=[0, 0, -5]),
        dict(top_scale=0.5, angles_top=[0, 0, 10], angles_bottom=[0, 0, -5]),
    ]
    
    probe_kwargs = probe_kwargs + [None] * 7
    colors = [myterial.blue_dark, myterial.pink_darker, myterial.deep_purple_darker]

    num_samples = [79, 79]
    interactive = False
    block_size_sub = 10
    simulation_kwargs = dict(max_depth=1, num_iters=10000, num_cpus=1)
    mouse_region_exp_probes(
        regions,
        num_samples,
        colors=colors,
        interactive=interactive,
        block_size_sub=block_size_sub,
        vis_only=vis_only,
        probe_kwargs=probe_kwargs,
        **simulation_kwargs,
    )
    plot_region_vals(
        load_df("mouse_region_exp_probes.csv"),
        "mouse_region_exp.pdf",
        x_name="Regions",
        scale=(12, 5)
    )


@app.command()
def do_hippocampus(ca1_ca3: bool = True, ca1_sub: bool = True):
    # CA3 CA1 figures

    # Uses tetrode_ca3_ca1_full
    if ca1_ca3:
        store_tetrode_results_depth()
        plot_samples_v_prop(
            load_df("samples_depth_ca3_ca1.csv"), "ca3_ca1_samps_depth.pdf"
        )

        store_npix_results()
        plot_pmf(load_df("npix_man.csv"), "npix_pmf.pdf")

    # CA1 SUB figures
    if ca1_sub:
        kwargs = {
            "num_iters": 1000,
            "do_graph": False,
            "use_mean": True,
            "num_graphs": 0,
            "sr": 0.01,
            "clt_start": 10,
            "fin_depth": 3,
        }
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_high.cfg"), "hc_high", **kwargs
        )
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_high_out.cfg"), "hc_high_out", **kwargs
        )
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_low.cfg"), "hc_low", **kwargs
        )
        connections_dependent_on_samples(
            parse_cfg("ca1_sub_vhigh.cfg"), "hc_vhigh", **kwargs
        )
        df_list = [
            load_df("connection_samples_hc_vhigh.csv"),
            load_df("connection_samples_hc_high.csv"),
            load_df("Connection_samples_hc_high_out.csv"),
            load_df("connection_samples_hc_low.csv"),
        ]
        df_names = [
            "2.8% (90% to 3.1%)",
            "1.8% (90% to 2%)",
            "1.8% (36% to 5%)",
            "0.8% (70% to 1%)",
        ]
        plot_exp_comp(
            df_list,
            df_names,
            "samples_hc_both.pdf",
            prop=True,
        )
        store_sub_results()
        dfs = [
            load_df("20_sub_vhigh.csv"),
            load_df("20_sub_high.csv"),
            load_df("20_sub_out.csv"),
            load_df("20_sub_low.csv"),
        ]
        plot_pmf_comp(dfs, df_names, "ca1_sub_tet_comp.pdf")


@app.command()
def produce_figures(
    figure_1: bool = True,
    figure_2: bool = True,
    figure_3: bool = True,
    figure_4: bool = True,
):
    if figure_1:
        figure1()
    if figure_2:
        figure2()
    if figure_3:
        figure3()
    if figure_4:
        figure4()


@app.command()
def figure1():
    do_explain()


@app.command()
def figure2():
    do_sub(do_full_vis=True)


@app.command()
def figure3():
    do_mouse_regions(vis_only=False)


@app.command()
def figure4():
    do_hippocampus()


@app.command()
def do_all(
    mouse: bool = True,
    explain: bool = True,
    accuracy: bool = True,
    examples: bool = True,
):
    """Produce all figures for the paper."""
    print("Reproducing all figures")

    if explain:
        do_explain(do_vis=True, do_pmf=True, do_dist=True)
    if examples:
        do_examples(do_exp=True, do_pmf=True, do_types=True)
    if mouse:
        do_mouse(do_mat_vis=True, do_comp=True, do_exp=True)
    if accuracy:
        do_accuracy(
            do_samples=True,
            do_pmf=True,
            do_regions=True,
            do_mouse_acc=True,
            do_exp=True,
            do_growth=True,
        )


if __name__ == "__main__":
    # app()
    do_mouse_regions(True)
    
