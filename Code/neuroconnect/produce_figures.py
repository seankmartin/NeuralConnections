"""Produce all the paper figures."""
import os
from configparser import ConfigParser
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .matrix import main as mouse_main
from .compound import (
    connections_dependent_on_samples,
    proportion,
    pmf_accuracy,
    connections_dependent_on_regions,
    distance_dependent_on_regions,
    mouse_region_exp,
    out_exp,
    explain_calc,
)
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
)
from .stats_convergence_rate import (
    test_network_convergence,
    test_config_convergence,
    test_rand_network_convergence,
)
from .stored_results import (
    store_region_results,
    store_tetrode_results,
    store_npix_results,
    store_sub_results,
)

here = os.path.dirname(os.path.realpath(__file__))


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


def do_accuracy(
    do_samples=True,
    do_pmf=True,
    do_regions=True,
    do_mouse_acc=True,
    do_exp=True,
    do_growth=True,
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
        ctrl_main(parse_cfg("recurrent_fig1.cfg"), args)

    if do_pmf:
        proportion(
            parse_cfg("recurrent_fig1.cfg"),
            depths=[
                1,
            ],
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


def do_all(mouse=True, explain=True, accuracy=True, examples=True):
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


def main():
    """Main entry point - seeds numpy before all figures generated."""
    np.random.seed(42)
    do_all(
        mouse=True,
        explain=True,
        accuracy=True,
        examples=True,
    )
    return


if __name__ == "__main__":
    main()
