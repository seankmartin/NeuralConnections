"""
Experimenting with assorted code in this package.

Can run with

python -m neuroconnect.playground
"""

import os
from collections import OrderedDict
from cv2 import mean

import numpy as np
import matplotlib.pyplot as plt
import mpmath
import seaborn as sns
import pandas as pd


def test_hyper_eg(total, bad, draws):
    """Demonstrates that expected overlap between two sets is formula."""
    from .connect_math import (
        hypergeometric_pmf,
        expected_non_overlapping,
        expected_overlapping,
    )

    exp = 0
    for k in range(draws + 1):
        res = hypergeometric_pmf(total, total - bad, draws, k)
        exp = exp + (k * res)

    other = expected_non_overlapping(total, bad, draws)
    other2 = expected_overlapping(total, bad, draws)
    return (exp, other, other2)


def test_unique(total, draws, n_senders):
    """Test the calculation of expected unique."""
    from .connect_math import expected_unique
    from .monte_carlo import monte_carlo

    total_a = np.array([i for i in range(total)])

    def gen_random(i):
        random = np.random.choice(total_a, draws)
        return (random,)

    def check_random(random):
        unique = np.unique(random)
        return (len(unique),)

    result = monte_carlo(check_random, gen_random, 50000)
    avg = 0
    for val in result:
        avg += val[0] / 50000
    ab = expected_unique(total, draws)

    print("Stats", ab, "MC", avg)

    senders = np.random.choice(total, size=(n_senders, draws), replace=True)
    connections = {}
    for i in range(n_senders):
        num_choices = np.random.randint(1, draws + 1, dtype=np.int32)
        forward_connection = senders[i, :num_choices]
        connections[i] = forward_connection

    avg = 0
    for k, v in connections.items():
        avg += len(np.unique(v)) / n_senders

    connections = 0
    for val in range(1, 301):
        connections += expected_unique(1000, val) / draws
    print("Stats", connections, "MC", avg)


def test_uniform(
    n_dists,
    min_val,
    max_val,
    n_dists2=20,
    n_iters=100000,
    N=1000,
    plot=True,
    n_senders=200,
):
    """Test the distribution of uniform distribution sums and functions of this."""
    # NOTE: of course, sum of uniform dists approaches normal dist
    # However, taking a function of the sum of uniform dists not necessarily

    from .connect_math import (
        nfold_conv,
        create_uniform,
        get_dist_mean,
        expected_unique,
        apply_fn_to_dist,
        alt_way,
    )
    from .monte_carlo import get_distribution
    from .mpf_connection import CombProb

    alt_full, alt_final = alt_way(N, n_dists, n_senders, min_val, max_val)
    uni = create_uniform(min_val, max_val)
    dists = [
        uni,
    ] * n_dists
    dist = nfold_conv(dists)

    print("Expected value: {}".format(get_dist_mean(dist)))

    def fn_to_apply(k):
        return float(expected_unique(N, k))

    fn_dist = apply_fn_to_dist(dist, fn_to_apply)

    print("Expected value fn: {}".format(get_dist_mean(fn_dist)))
    print("Old expected value dist: {}".format(expected_unique(N, get_dist_mean(dist))))
    print(
        "Old expected value: {}".format(
            expected_unique(N, n_dists * (max_val + min_val) / 2)
        )
    )

    randoms = np.random.randint(min_val, max_val + 1, size=(n_iters, n_dists))
    good = np.array([i for i in range(n_senders)])
    choices = np.array([i for i in range(N)])
    random_vals = np.random.choice(
        choices, size=(n_iters, n_dists, max_val), replace=True
    )
    values = []
    for i in range(n_iters):
        arrs = []
        for j in range(n_dists):
            arrs.append(random_vals[i, j, : randoms[i, j]])

        to_use = np.concatenate(arrs)
        drawn = np.random.choice(choices, size=(n_dists2,), replace=False)

        values.append(
            [len(np.unique(to_use)), len(to_use), len(np.intersect1d(drawn, to_use))]
        )

    df = pd.DataFrame(values, columns=["Unique", "Samples", "Overlap"])
    old_dist = get_distribution(df, "Samples", n_iters)

    values = []
    for i in range(n_iters):
        arrs = []
        # Can remove this to just test delta - n_dists_use = n_dists
        send_drawn = np.random.choice(choices, size=(n_dists,), replace=False)
        n_dists_use = len(np.intersect1d(send_drawn, good))
        if n_dists_use == 0:
            values.append([0, 0, 0])
            continue

        for j in range(n_dists_use):
            arrs.append(random_vals[i, j, : randoms[i, j]])

        to_use = np.concatenate(arrs)
        drawn = np.random.choice(choices, size=(n_dists2,), replace=False)

        values.append(
            [len(np.unique(to_use)), len(to_use), len(np.intersect1d(drawn, to_use))]
        )
    df = pd.DataFrame(values, columns=["Unique", "Samples", "Overlap"])
    est_dist = get_distribution(df, "Overlap", n_iters)
    new_dist = get_distribution(df, "Unique", n_iters)

    new_fn_dist = OrderedDict()
    for k in new_dist.keys():
        new_fn_dist[k] = fn_dist.get(k, 0)

    if plot:
        from .plot import plot_distribution

        plot_distribution(dist, "uniform_test.png")
        plot_distribution(new_fn_dist, "uniform_fn_test.png")
        plot_distribution(old_dist, "uniform_norm_test.png")
        plot_distribution(new_dist, "uniform_sim_test.png")
        plot_distribution(est_dist, "uniform_fin_test.png")
        plot_distribution(alt_full, "uniform_to_test.png")
        plot_distribution(alt_final, "uniform_fin_alt.png")

        fig, ax = plt.subplots()
        dists = (est_dist, stats, alt_final)
        names = ("Simulated", "Old stats", "New stats")
        colors = ("b", "r", "g")
        for i, (dist, name) in enumerate(zip(dists, names)):
            x = list(dist.keys())
            y = list(dist.values())
            ax.plot(x, y, c=colors[i], label=name)
        plt.legend()
        here = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(os.path.join(here, "..", "figures"), exist_ok=True)
        fig.savefig(os.path.join(here, "..", "figures", "un_all.png"))
        plt.close("all")

    print(get_dist_mean(dist), get_dist_mean(old_dist))
    print(
        get_dist_mean(fn_dist),
        expected_unique(N, n_dists * (max_val + min_val) / 2),
    )
    print(get_dist_mean(est_dist), get_dist_mean(stats), get_dist_mean(alt_final))
    print(get_dist_mean(new_dist), get_dist_mean(alt_full))

    return dist


def test_sample_graph():
    """Test calculations of paths for a small known graph."""
    from .simple_graph import vis_graph, find_connected_limited
    from .nx_graph import nx_create_graph, nx_vis_force, nx_find_connected_limited

    graph = [[] for _ in range(12)]

    graph[0] = [3]
    graph[1] = [8]
    graph[2] = [1]
    graph[3] = [6]
    graph[4] = [8]
    graph[5] = [9]
    graph[6] = [5]
    graph[7] = [8]
    graph[8] = [7]
    graph[9] = [11]
    graph[10] = [11]
    graph[11] = [10]

    nx_graph = nx_create_graph(graph)

    start = [0, 1, 2, 3, 4, 5]
    end = [6, 7, 8, 9, 10, 11]

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "..", "figures"), exist_ok=True)
    # nx_vis_force(nx_graph, start, end, [], [], "hand_nx.png", True)
    fig = vis_graph(graph, [6, 6], start, end)
    fig.savefig(os.path.join(here, "..", "figures", "hand_simple.png"))

    for depth in range(4):
        simple = find_connected_limited(graph, start, end, depth)
        nx = nx_find_connected_limited(nx_graph, start, end, depth)
        print(depth, simple, nx)

    new_start = [0, 2]

    for depth in range(7):
        simple = find_connected_limited(graph, new_start, end, depth)
        nx = nx_find_connected_limited(nx_graph, new_start, end, depth)
        print(depth, simple, nx)


def sample_graph_exp():
    """Test calculations of paths for a small known graph."""
    from .simple_graph import vis_graph, find_connected_limited
    from .nx_graph import nx_create_graph, nx_vis_force, nx_find_connected_limited
    from .monte_carlo import (
        monte_carlo,
        list_to_df,
        summarise_monte_carlo,
        get_distribution,
    )

    graph = [[] for _ in range(14)]

    graph[0] = [7]
    graph[1] = [3]
    graph[2] = []
    graph[3] = [9]
    graph[4] = [12, 5]
    graph[5] = [4]
    graph[6] = [13]
    graph[7] = []
    graph[8] = []
    graph[9] = []
    graph[10] = [8, 9]
    graph[11] = []
    graph[12] = []
    graph[13] = [4]

    connected = [0, 3, 4, 6]
    fixed_samples = 2
    num_sampled = 3

    start = [0, 1, 2, 3, 4, 5, 6]
    source_vert_list = np.array(start)
    end = [7, 8, 9, 10, 11, 12, 13]
    targets = np.array(end)

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "..", "figures"), exist_ok=True)
    fig = vis_graph(graph, [7, 7], start, end)
    fig.savefig(os.path.join(here, "..", "figures", "hand_simple_2.png"))

    # def random_var_gen():
    #     sources = np.append(
    #         np.random.choice(connected, fixed_samples, replace=False),
    #         np.random.choice(
    #             np.delete(source_vert_list, connected),
    #             num_sampled - fixed_samples,
    #             replace=False,
    #         ),
    #     )

    #     return (sources,)

    # def fn_to_eval(sources):
    #     reachable = find_connected_limited(graph, sources, targets, max_depth=1)
    #     # reachable = find_connected(graph, sources, targets)
    #     return (len(reachable),)

    def random_var_gen(iter_val):
        sources = np.random.choice(source_vert_list, num_sampled, replace=False)
        targets_t = np.random.choice(targets, num_sampled, replace=False)

        return (sources, targets_t)

    def fn_to_eval(sources, targets_t):
        reachable = find_connected_limited(graph, sources, targets_t, max_depth=3)
        return (len(reachable),)

    result = monte_carlo(fn_to_eval, random_var_gen, 10000, num_cpus=1)
    df = list_to_df(
        result,
        [
            "Connections",
        ],
    )
    result = summarise_monte_carlo(
        df,
        to_plot=[
            "Connections",
        ],
        plt_outfile="graph_dist.png",
        plot=False,
    )
    distrib = get_distribution(df, "Connections", 10000)

    fig, ax = plt.subplots(figsize=(6, 8))
    x = np.array(list(distrib.keys()), dtype=float)
    y = np.array(list(distrib.values()), dtype=float)

    ax.plot(x, y, "ko", ms=2.5)
    y_vals_min = [0 for _ in x]
    y_vals_max = y
    colors = ["k" for _ in x]
    ax.vlines(x, y_vals_min, y_vals_max, colors=colors)
    sns.despine()
    plt.xlabel("Number of recorded connections")
    plt.ylabel("Probability of occurrence")
    fig.savefig(os.path.join(here, "..", "figures", "pdf_small_3.pdf"), dpi=400)

    return {"distrib": distrib, "summary": result, "full": df}


def plot_expected(input_file):
    """The change in value of expected is exponential."""
    x, y = [], []
    y2 = []
    with open(input_file, "r") as f:
        contents = f.read().split(",")
        for line in contents:
            num, mp = line.split(":")
            num = int(num)
            mp = float(mp[6:-2])
            x.append(num)
            y.append(mp)
            y2.append(234 + (1 - np.exp(-0.3 * num)) * 766)

    plt.plot(x, y, c="k")
    plt.plot(x, y2, c="b")
    plt.savefig("test.png")


def check_subsample():
    """Check the accuracy of subsampling distributions."""
    from .connect_math import create_normal, get_uniform_moments
    import matplotlib.pyplot as plt
    from time import time

    min_val, max_val = 100, 3000
    mean, var = get_uniform_moments(min_val, max_val)
    mult = 100

    st = time()
    normal_1 = create_normal(
        range((max_val * mult) + 1), mean * mult, var * mult, sub=None
    )
    print("took {:.2f} seconds".format(time() - st))

    st = time()
    normal_2 = create_normal(
        range((max_val * mult) + 1), mean * mult, var * mult, sub=0.01
    )
    print("took {:.2f} seconds".format(time() - st))

    fig, ax = plt.subplots()

    x, y = (list(normal_1.keys()), list(normal_1.values()))
    ax.plot(x, y, "r", label="full")
    x, y = (list(normal_2.keys()), list(normal_2.values()))
    ax.plot(x, y, "b", linestyle="--", label="sample")

    plt.show()


def test_conv_speed(min_val, max_val, clt_start1, clt_start2, max_samples):
    """Test the speed of the convolution operation (hint - it is slow)."""
    from time import time

    from .connect_math import (
        create_uniform,
        convolution,
        create_normal,
        get_dist_mean,
        get_dist_var,
    )

    dist = create_uniform(min_val, max_val)
    start_mean, start_var = get_dist_mean(dist), get_dist_var(dist)
    start_max_val = max(list(dist.keys()))
    cache = OrderedDict()
    cache[0] = OrderedDict()
    cache[0][0] = 1.0

    start1 = time()
    for i in range(1, max_samples + 1):
        if i < clt_start1:
            cache[i] = convolution(dist, cache[i - 1])
        else:
            cache[i] = create_normal(
                range((start_max_val * i) + 1), start_mean * i, start_var * i
            )

    end1 = time()

    cache = OrderedDict()
    cache[0] = OrderedDict()
    cache[0][0] = 1.0
    start2 = time()

    for i in range(1, max_samples + 1):
        if i < clt_start2:
            cache[i] = convolution(dist, cache[i - 1])
        else:
            cache[i] = create_normal(
                range((start_max_val * i) + 1), start_mean * i, start_var * i
            )

    end2 = time()

    print(
        "Completed {} samples in {:.2f} and {:.2f} seconds with {}, and {}".format(
            max_samples, end1 - start1, end2 - start2, clt_start1, clt_start2
        )
    )


def vis_graph_as_matrix():
    from .simple_graph import to_matrix, matrix_vis, create_graph
    from .matrix import gen_random_matrix
    from .connectivity_patterns import get_by_name

    mat = gen_random_matrix(1000, 1500, 0.01, 0.05, 0.001, 0.005)
    matrix_vis(*mat, 10, name="random_mat_vis.png")

    connect_pattern = get_by_name("mean_connectivity")
    a_params = dict(
        num_senders=100,
        min_inter=0.0005,
        max_inter=0.002,
        min_forward=100,
        max_forward=200,
    )
    b_params = dict(
        num_senders=200,
        min_inter=0.003,
        max_inter=0.007,
        min_forward=150,
        max_forward=350,
    )
    graph, _ = create_graph([1000, 1500], connect_pattern, [a_params, b_params])
    mat_graph = to_matrix(graph, 1000, 1500)
    matrix_vis(*mat_graph, 10, name="random_graph_vis.png")

    # TODO add in the connect matrix strat in block


def connect_prob_large_matrix():
    from .matrix import convert_mouse_data, load_matrix_data
    from .atlas import gen_graph_for_regions
    from .mpf_connection import CombProb
    from .connectivity_patterns import MatrixConnectivity
    from .plot import plot_pmf_accuracy
    from skm_pyutils.py_table import list_to_df, df_to_file, df_from_file
    from pathlib import Path
    import pickle
    import sys
    import numpy as np
    from pprint import pprint
    from time import perf_counter

    np.random.seed(42)
    here = os.path.dirname(os.path.abspath(__file__))
    A_name, B_name = ["VISp", "VISl"]
    region_sizes = [333055, 49569]
    num_sampled = [2, 2]
    max_depth = 2
    to_use = [True, True, True, True]
    mean_estimate = False
    force_no_mean = True
    sr = 0.01
    clt_start = 30
    main_dir = Path(__file__).resolve().parent.parent
    pickle1 = main_dir / "pickles" / "region_points.pkl"
    pickle2 = main_dir / "pickles" / "stats.pkl"

    pickle1.parent.mkdir(exist_ok=True)

    t1 = perf_counter()

    if os.path.isfile(pickle1):
        a_indices, b_indices = pickle.load(open(pickle1, "rb"))
        args_dict = pickle.load(open(pickle2, "rb"))
        print(
            "{} - {}, {} - {}".format(
                A_name, B_name, args_dict["num_start"], args_dict["num_end"]
            )
        )
        print("inside of probe are {} - {}".format(len(a_indices), len(b_indices)))
    else:
        convert_mouse_data(A_name, B_name)
        mc, args_dict = load_matrix_data(to_use, A_name, B_name)
        print("{} - {}, {} - {}".format(A_name, B_name, mc.num_a, mc.num_b))

        region_pts = gen_graph_for_regions(
            (A_name, B_name), region_sizes, sort_=True, shift=True
        )[0]
        a_indices = region_pts[0][1]
        b_indices = region_pts[1][1]

        ## This is mpf_probe_connectome
        probe_stats = mc.compute_probe_stats(
            a_indices,
            b_indices,
        )
        sub_args_dict = probe_stats["stats"]
        for k, v in sub_args_dict.items():
            args_dict[f"{k}_probe"] = v
        sub_args_dict = probe_stats["A_stats"]
        for k, v in sub_args_dict.items():
            args_dict[f"{k}_A"] = v
        sub_args_dict = probe_stats["B_stats"]
        for k, v in sub_args_dict.items():
            args_dict[f"{k}_B"] = v
        sub_args_dict = probe_stats["inter"]
        for k, v in sub_args_dict.items():
            args_dict[k] = v
        args_dict["total_samples"] = num_sampled[0]

        pickle.dump((a_indices, b_indices), file=open(pickle1, "wb"))
        pickle.dump(args_dict, file=open(pickle2, "wb"))
        print("inside of probe are {} - {}".format(len(a_indices), len(b_indices)))

    args_dict["static_verbose"] = True
    args_dict["max_depth"] = max_depth
    args_dict["clt_start"] = clt_start
    args_dict["mean_estimate"] = mean_estimate
    args_dict["total_samples"] = num_sampled[0]
    if max_depth > 1:
        sr = 0.01
    if mean_estimate is True:
        sr = None

    if force_no_mean:
        args_dict["use_mean"] = False
    cp = CombProb(
        args_dict["num_start_probe"],
        num_sampled[0],
        args_dict["num_senders_probe"],
        args_dict["num_end_probe"],
        num_sampled[1],
        MatrixConnectivity.static_expected_connections,
        verbose=True,
        subsample_rate=sr,
        **args_dict,
    )

    # print(sys.getsizeof(args_dict))
    # print(sys.getsizeof(region_pts))
    t2 = perf_counter()
    t_taken = t2 - t1
    print(f"Took {t_taken:.2f} seconds")

    result = {
        "expected": cp.expected_connections(),
        "total": cp.get_all_prob(),
        "each_expected": {k: cp.expected_total(k) for k in range(num_sampled[0] + 1)},
    }

    with open(os.path.join(here, "..", "results", "test_prob.txt"), "w") as f:
        pprint(result, width=120, stream=f)

    l = []
    for k, v in result["total"].items():
        l.append([k, v, "Statistical estimation"])
    df = list_to_df(
        l, headers=["Number of connected neurons", "Probability", "Calculation"]
    )
    df_to_file(df, os.path.join(here, "..", "results", "test_csv.csv"))

    to_load = os.path.join(here, "..", "results", "test_pmf.csv")
    if os.path.isfile(to_load):
        plot_pmf_accuracy(df_from_file(to_load), "test_pmf.pdf")


def test_depth_2():
    from .connectivity_patterns import RecurrentConnectivity, MatrixConnectivity
    from .simple_graph import create_graph, reverse, to_matrix
    from .matrix import graph_connectome, mpf_probe_connectome
    from pprint import pprint
    import numpy as np

    np.random.seed(42)

    region_sizes = [600, 800]
    num_sampled = [6, 8]
    max_depth = 3
    mean_estimate = True

    out_kwargs = dict(
        num_senders=180,
        min_inter=0.02,
        max_inter=0.07,
        min_forward=4,
        max_forward=10,
    )

    recur_kwargs = dict(
        num_senders=200,
        min_inter=0.01,
        max_inter=0.05,
        min_forward=3,
        max_forward=8,
    )

    connectivity_params = [out_kwargs, recur_kwargs]

    g_graph, g_connected = create_graph(
        region_sizes, RecurrentConnectivity, connectivity_params
    )
    r_graph = reverse(g_graph)

    a_indices = [i for i in range(70)]
    b_indices = [i for i in range(10, 45)]

    full_res = {}

    full_res["graph"] = graph_connectome(
        num_sampled, 
        max_depth,
        num_iters=10000,
        graph=g_graph,
        reverse_graph=r_graph,
        to_write=region_sizes,
        a_indices=a_indices,
        b_indices=b_indices,
    )

    AB, BA, AA, BB = to_matrix(g_graph, *region_sizes)
    mc = MatrixConnectivity(ab=AB, ba=BA, bb=BB, aa=AA)
    full_res["stats"] = mpf_probe_connectome(
        mc,
        num_sampled,
        a_indices,
        b_indices,
        max_depth,
        args_dict=mc.compute_stats(),
        mean_estimate=mean_estimate,
        force_no_mean=mean_estimate,
        sr=None,
    )

    pprint(full_res)


if __name__ == "__main__":
    # Hypergeometric again right?
    # connect_prob_large_matrix()
    test_depth_2()
    exit(-1)

    arr = [
        405389,
        375280,
        390413,
        404149,
        377923,
        439143,
        431692,
        383157,
        372278,
        388033,
        394696,
        355324,
        376727,
        347152,
        388628,
        361182,
        400800,
        353701,
        449542,
        424401,
        417760,
        448637,
        429387,
        355282,
        415545,
        453806,
        356367,
        398820,
        377551,
        420042,
        437768,
        419263,
        419257,
        405574,
        445007,
        362068,
        368380,
        350797,
        359602,
        392117,
        366477,
        428624,
        384986,
        372038,
        439853,
        348657,
        442893,
    ]

    print(len(arr))

    print(test_hyper_eg(1000, 50, 10))
    print(test_hyper_eg(1000, 100, 100))

    if os.path.isfile("/home/sean/Repos/privateCode/ConnectionsCode/exp.txt"):
        plot_expected("/home/sean/Repos/privateCode/ConnectionsCode/exp.txt")

    p0 = mpmath.mpf(56 / 90)
    p1 = mpmath.mpf(32 / 90)
    p2 = mpmath.mpf(2 / 90)

    exp1 = np.array([p0, p1, p2])

    iters = 10000
    vals = [i for i in range(10)]

    a = np.array([0, 0, 0])
    for i in range(iters):
        total_overlaps = 0
        list_1 = np.random.choice(vals, size=2, replace=False)
        list_2 = np.random.choice(vals, size=2, replace=False)
        for l in list_1:
            if l in list_2:
                total_overlaps += 1
        a[total_overlaps] += 1

    exp2 = a / iters

    print(exp1, exp2)

    print(np.sum(exp1 * np.array([4, 3, 2])))
    print(np.sum(exp2 * np.array([4, 3, 2])))

    def C(n, r):
        return mpmath.binomial(n, r)

    print(C(4, 2))
    print(C(3.5, 2))

    print(sample_graph_exp())
    print(test_sample_graph())

    vis_graph_as_matrix()
