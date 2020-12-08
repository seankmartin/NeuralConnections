"""Some small extra utility that shouldn't be in cli.py"""

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import mpmath

from neuroconnect.playground import (
    test_sample_graph,
    sample_graph_exp,
    test_uniform,
    test_conv_speed,
    check_subsample,
)
from neuroconnect.connect_math import (
    create_normal,
    get_uniform_moments,
    get_dist_mean,
    get_dist_var,
    create_uniform,
    nfold_conv,
    hypergeometric_pmf,
)
from neuroconnect.stored_results import main as store
from neuroconnect.plot import main as to_plot


def example():
    k3 = OrderedDict()
    k3[0] = (3 / 7) * (2 / 6) * (1 / 5)
    k3[1] = 3 * (4 / 7) * (3 / 6) * (2 / 5)
    k3[2] = 3 * (4 / 7) * (3 / 6) * (3 / 5)
    k3[3] = (4 / 7) * (3 / 6) * (2 / 5)

    print(k3)

    def delta(k):
        return float(hypergeometric_pmf(7, 4, 3, k))

    k2 = OrderedDict()
    for i in range(4):
        k2[i] = delta(i)

    print(k2)

    def gamma(x, l):
        return hypergeometric_pmf(7, l, 3, x)

    def full(x, l):
        return k3.get(l, 0) * gamma(x, l)

    final_dist = OrderedDict()
    final_dist[0] = full(0, 0) + full(0, 1) + full(0, 2) + full(0, 3)
    final_dist[1] = full(1, 1) + full(1, 2) + full(1, 3)
    final_dist[2] = full(2, 2) + full(2, 3)
    final_dist[3] = full(3, 3)

    print(np.sum(np.array(list(final_dist.values()))))
    return final_dist


if __name__ == "__main__":
    np.random.seed(42)

    store()
    to_plot()
    exit(-1)

    print(example())
    exit(-1)

    actual = hypergeometric_pmf(1000, 200, 20, 5)
    estimated = hypergeometric_pmf(1000, 200, 20, 5, True)
    print(actual, estimated)
    exit(-1)

    check_subsample()
    exit(-1)

    test_conv_speed(1, 300, 30, 10, 100)
    exit(-1)

    mean, var = get_uniform_moments(50, 250)
    unif = create_uniform(50, 250)
    total = 15
    normal = create_normal(range((total * 250) + 1), mean * total, var * total)
    unif_conv = nfold_conv([unif] * total)

    plt.plot(
        list(normal.keys()), list(normal.values()), label="norm", linestyle="--", c="k"
    )
    plt.plot(list(unif_conv.keys()), list(unif_conv.values()), label="unif")
    plt.legend()
    plt.show()

    exit(-1)

    test_sample_graph()

    print(sample_graph_exp())

    test_uniform(20, 50, 250)
