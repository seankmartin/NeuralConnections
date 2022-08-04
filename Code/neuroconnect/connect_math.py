"""Basic maths functions related to graph connection probability."""
from math import ceil, floor, sqrt, isclose
from scipy import sparse, stats, interpolate
from collections import OrderedDict

import mpmath
import numpy as np

from .plot import plot_acc_interp


def C(n, r):
    """Implement n choose r, works on non integers via gamma."""
    return mpmath.binomial(n, r)


def binomial_pmf(n, k, p):
    """Binomial: n draws, k successes, and p proportion successes."""
    bin_res = C(n, k) * mpmath.power(p, k) * mpmath.power(1 - p, n - k)
    return bin_res


def hypergeometric_pmf(N, K, n, k, approx=False):
    """
    Calculates the hypergeometric probability mass function.

    See https://en.wikipedia.org/wiki/Hypergeometric_distribution

    Parameters
    ----------
    N : float
        The population size.
    K : float
        The number of success states in the population.
    n : int
        The number of draws.
    k : int
        The number of observed successes.
    approx : bool
        If True, approximate the result by the binomial distribution.

    Returns
    -------
    float
        The pmf pX(k).

    """
    if approx is True:
        proportion_true = K / N
        return binomial_pmf(n, k, proportion_true)

    left = C(K, k)
    right = C(N - K, n - k)
    btm = C(N, n)
    return left * right / btm


def expected_unique(N, k, do_round=True):
    """
    Calculates the expected number of unique values when drawing k from N.

    See https://math.stackexchange.com/questions/72223/finding-expected-number-of-distinct-values-selected-from-a-set-of-integers

    Parameters
    ----------
    N : int
        The population size.
    k : int
        The number of draws.

    Returns
    -------
    float
        The expected number of distinct values.

    """
    n = mpmath.mpf(N)
    p = mpmath.mpf(k)
    temp = mpmath.power((n - 1) / n, p)
    if do_round:
        return max(int(round(n * (1 - temp))), 0)
    else:
        return max(float(n * (1 - temp)), 0)


def expected_overlapping(total, good, draws):
    """Calculates the expected overlapping draws from total of good."""
    if total == 0:
        return 0
    drawn = int(round(draws * (good / total)))
    return min(max(drawn, 0), int(round(good)))


def expected_non_overlapping(total, bad, draws):
    """Calculates the expected non-overlapping draws from total of bad."""
    good = total - bad
    return expected_overlapping(total, good, draws)


def window_2d_avg(matrix, k_size):
    """Get the average of matrix in k_size * k_size windows."""
    x, y = matrix.shape
    add_val = 1 / (k_size * k_size)

    new_x = int(ceil(x / k_size))
    new_y = int(ceil(y / k_size))

    out = np.zeros(shape=(new_x, new_y), dtype=np.float32)

    nonzeros_x, nonzeros_y, _ = sparse.find(matrix)
    for i, j in zip(nonzeros_x, nonzeros_y):
        x_idx = int(floor(i / k_size))
        y_idx = int(floor(j / k_size))
        out[x_idx, y_idx] += add_val

    # for i in range(new_x):
    #     for j in range(new_y):
    #         out[i, j] = np.mean(
    #             matrix[i * k_size : (i + 1) * k_size, j * k_size : (j + 1) * k_size]
    #         )

    return out


def multi_2d_avg(AB, BA, AA, BB, k_size):
    """Combine separate into one sparse matrix."""
    x, y = AB.shape

    a_size = int(ceil(x / k_size))
    b_size = int(ceil(y / k_size))

    full_mat = np.zeros(shape=(a_size + b_size, a_size + b_size), dtype=np.float32)
    full_mat[:a_size, :a_size] = window_2d_avg(AA, k_size)
    full_mat[:a_size, a_size:] = window_2d_avg(AB, k_size)
    full_mat[a_size:, :a_size] = window_2d_avg(BA, k_size)
    full_mat[a_size:, a_size:] = window_2d_avg(BB, k_size)

    return full_mat


def convolution(X, Y, sub=None):
    """Computes Z = X + Y as distributions."""
    min_val = np.array(list(X.keys())).min() + np.array(list(Y.keys())).min()
    max_val = np.array(list(X.keys())).max() + np.array(list(Y.keys())).max()

    out_dict = OrderedDict()
    r1 = range(min_val, max_val + 1)
    first_iter, s1 = subsample_list(r1, sub)
    for i in first_iter:
        out_dict[i] = 0
        r2 = range(0, i + 1)
        if len(r1) > 1:
            second_iter, s2 = subsample_list(r2, sub)
        else:
            second_iter, s2 = r2, False
        values = np.zeros(len(second_iter))
        for j_idx, j in enumerate(second_iter):
            val = X.get(j, 0) * Y.get(i - j, 0)
            values[j_idx] = val
        if s2:
            values = interp(list(r2), second_iter, values)
        out_dict[i] = np.sum(values)

    if s1:
        keys = list(r1)
        final_vals = interp(keys, list(out_dict.keys()), list(out_dict.values()))
        total = np.sum(final_vals)
        out_dict = OrderedDict()
        for k, v in zip(keys, final_vals):
            out_dict[k] = v / total

    return out_dict


def nfold_conv(args, sub=None):
    """Computes SUM(args) where args are distributions."""
    if len(args) >= 2:
        conv = convolution(args[0], args[1], sub=sub)
        for arg in args[2:]:
            conv = convolution(conv, arg, sub=sub)
        return conv
    else:
        return args[0]


def create_uniform(min_val, max_val):
    """Create a uniform distribution with min and max values."""
    dist = OrderedDict()

    div = 1 / (max_val + 1 - min_val)
    for val in range(min_val, max_val + 1):
        dist[val] = div

    return dist


def get_dist_mean(dist):
    """Get the expected value of a distribution."""
    mean = 0
    for k, v in dist.items():
        mean += k * v

    return mean


def get_dist_var(dist):
    """Get the variance of a distribution (OrderedDict)."""
    mean = get_dist_mean(dist)

    var = 0
    for k, v in dist.items():
        to_add = v * (k - mean) * (k - mean)
        var += to_add

    return var


def trim_zeros(dist):
    new_dist = OrderedDict()

    values = []
    for k, v in dist.items():
        if v != 0.0:
            values.append(k)

    if len(values) > 1:
        min_val = min(values)
        max_val = max(values)

        for i in range(min_val, max_val + 1):
            new_dist[i] = dist.get(i, 0)
    else:
        new_dist[values[0]] = dist.get(values[0], 0)

    return new_dist


def apply_fn_to_dist(dist, fn, sub=None):
    """Apply a function to the x values in a distribution - linearly interpolates."""
    new_dist = OrderedDict()

    if sub is not None:
        dist = trim_zeros(dist)
        keys = list(dist.keys())
        if not isinstance(keys[0], int):
            raise ValueError("Distribution must have integer keys, got {}".format(keys))
        keys, did_sub = subsample_list(keys, sub)
    else:
        keys = list(dist.keys())

    for k in keys:
        evaled = fn(k)
        v = dist[k]
        if evaled in new_dist:
            new_dist[evaled] += v
        else:
            new_dist[evaled] = v

    return dist_from_samples(new_dist)


def random_draw_dist(
    max_samples, dist, total, apply_fn=True, keep_all=False, clt_start=30, sub=None
):
    """Expected unique values with n_samples from dist from total possible."""

    def fn_to_apply(k):
        return float(expected_unique(total, k))

    sum_dist = None
    mean, var = get_dist_mean(dist), get_dist_var(dist)
    results = OrderedDict()
    for n_samples in range(0, int(max_samples) + 1):
        if n_samples == 0:
            if keep_all:
                to_out = OrderedDict()
                to_out[0] = 1
                results[n_samples] = to_out
            else:
                results[n_samples] = 0
            continue
        elif n_samples <= 2:
            dists_to_conv = [dist] * int(n_samples)
        else:
            dists_to_conv = [sum_dist, dist]
        if n_samples < clt_start:
            sum_dist = nfold_conv(dists_to_conv, sub=sub)
        else:
            max_val = max(list(dist.keys()))
            sum_dist = create_normal(
                range((int(max_val * n_samples)) + 1),
                mean * n_samples,
                var * n_samples,
                sub=sub,
                interp_after=True,
            )
        if apply_fn is False:
            results[n_samples] = sum_dist
            continue
        fn_dist = apply_fn_to_dist(sum_dist, fn_to_apply)
        if keep_all:
            results[n_samples] = fn_dist
        else:
            results[n_samples] = get_dist_mean(fn_dist)

    return results


def interp(x_samps, xvals, yvals):
    """Interpolate linearly between xvals and yvals at x_samps."""
    if not np.all(np.diff(xvals) > 0):
        raise ValueError("Require monotonically increasing x values to interp")
    try:
        if len(xvals) == 1:
            yinterp = yvals
        kind = "linear"
        f = interpolate.interp1d(xvals, yvals, kind=kind)
        yinterp = f(x_samps)
    except BaseException:
        yinterp = np.interp(x_samps, xvals, yvals)

    return yinterp


def dist_from_samples(dist):
    """Perform linear interpolation to estimate a distribution from samples."""
    xvals = np.array(list(dist.keys()))
    yvals = np.array(list(dist.values()))
    min_val = int(floor(xvals.min()))
    max_val = int(ceil(xvals.max()))
    x_samps = np.array([i for i in range(min_val, max_val + 1)])
    if not np.all(np.diff(xvals) > 0):
        raise ValueError("Need monotonically increasing x points, got {}".format(xvals))

    yinterp = interp(x_samps, xvals, yvals)
    div_val = np.sum(yinterp)
    if div_val == 0:
        raise ValueError("Got zero probability from x-{}, y-{}".format(xvals, yvals))
    yinterp = yinterp / div_val

    new_dist = OrderedDict()
    for k, v in zip(x_samps, yinterp):
        new_dist[int(k)] = v

    return new_dist


def alt_way(N, samps, senders, min_val, max_val):
    """Alternative calculation of probability connection."""
    prob_a_senders = OrderedDict()

    for i in range(samps + 1):
        prob_a_senders[i] = float(hypergeometric_pmf(N, senders, samps, i))

    dist = create_uniform(min_val, max_val)
    dists = random_draw_dist(samps, dist, N, keep_all=True)

    to_eval = [i for i in range(N + 1)]
    weighted_dist = OrderedDict()
    for k in to_eval:
        weighted_dist[k] = 0

    for g in dists.keys():
        dist = dists[g]
        prob = prob_a_senders[g]
        for k in to_eval:
            weighted_dist[k] += prob * dist.get(k, 0)

    weighted_dist_vals = np.array(list(weighted_dist.values()))
    pdf_total = np.sum(weighted_dist_vals)
    if pdf_total - 1.0 > 0.0001:
        raise RuntimeError("PDF does not sum to 1, got {}.".format(pdf_total))

    final_res = OrderedDict()

    for i in range(samps + 1):

        def fn_to_apply(k):
            return float(hypergeometric_pmf(N, k, samps, i))

        hyper_weight = 0
        for k, v in weighted_dist.items():
            hyper_weight += fn_to_apply(k) * v
        final_res[i] = hyper_weight

    return weighted_dist, final_res


def second_deriv_improve_interp(x_samps, xvals, yvals, dist, fn_to_eval, x_val):
    """Given samples at xvals and yvals, second derivative to detect change points."""
    grad = np.gradient(yvals)
    diffs = np.abs(grad)
    grad_2 = np.gradient(grad)

    m = np.mean(diffs)
    s = np.std(diffs)
    ub = m + (3 * s)
    outliers = []
    for i, val in enumerate(diffs[:2]):
        if val >= ub:
            outliers.append(i)
            outliers.append(i + 1)
    for i, val in enumerate(diffs[-2:]):
        if val >= ub:
            outliers.append(i)
            outliers.append(i + 1)

    for i in range(len(grad_2) - 1):
        tc = abs(grad_2[i + 1] - grad_2[i])
        cv = np.abs((grad_2[i + 1] + grad_2[i]))
        if tc > cv and tc > (0.00000000001):
            outliers.append(i)

    outliers = sorted(list(set(outliers)))

    total_so_far = 0
    for val in outliers:
        start_idx = total_so_far + val
        new_x = np.array(
            list(range(xvals[start_idx] + 1, xvals[start_idx + 1])),
            dtype=np.int32,
        )
        new_y = np.zeros(
            shape=(len(new_x)),
            dtype=np.float64,
        )
        for i_idx, i in enumerate(new_x):
            v = dist[i]
            if not isclose(v, 0.0, abs_tol=1e-8):
                new_y[i_idx] = v * fn_to_eval(i, x_val)
            else:
                new_y[i_idx] = 0.0

        first_bit_x, second_bit_x = (
            xvals[: start_idx + 1],
            xvals[start_idx + 1 :],
        )
        first_bit_y, second_bit_y = (
            yvals[: start_idx + 1],
            yvals[start_idx + 1 :],
        )
        total_so_far += len(new_x)
        xvals = np.concatenate((first_bit_x, new_x, second_bit_x))
        yvals = np.concatenate((first_bit_y, new_y, second_bit_y))

    return xvals, yvals


def marginal_prob(
    eval_range,
    dist,
    fn_to_eval,
    x=None,
    sub=None,
    plot=False,
    true_plot=False,
    keep_all=False,
):
    """Marginal probability of dist over eval_range calculating fn."""
    final_res = OrderedDict()

    def x_prob(x_val):
        if sub is not None:
            in_sub = sub / 10
        else:
            in_sub = sub

        keys, did_sub = subsample_list(dist.keys(), in_sub)
        weighted_prob = np.zeros(shape=len(keys), dtype=np.float64)
        for i, k in enumerate(keys):
            v = dist[k]
            if not isclose(v, 0.0, abs_tol=1e-8):
                weighted_prob[i] = v * fn_to_eval(k, x_val)

        if did_sub:
            x_samps = list(dist.keys())
            xvals = keys
            yvals = weighted_prob
            xvals, yvals = second_deriv_improve_interp(
                x_samps, xvals, yvals, dist, fn_to_eval, x_val
            )
            weighted_prob = interp(x_samps, xvals, yvals)
            weighted_prob[weighted_prob < 0] = 0.0

            if plot is True:
                if true_plot is True:
                    keys, did_sub = subsample_list(dist.keys(), None)
                    true_y = np.zeros(shape=len(keys), dtype=np.float64)
                    for i, k in enumerate(keys):
                        v = dist[k]
                        true_y[i] = v * fn_to_eval(k, x_val)
                else:
                    true_y = None
                plot_acc_interp(
                    x_samps,
                    weighted_prob,
                    xvals,
                    yvals,
                    "dist_marg_{}.pdf".format(x_val),
                    true_y=true_y,
                )

        if keep_all:
            od = OrderedDict()
            for k, v in zip(dist.keys(), weighted_prob):
                od[k] = v
            return od
        else:
            return np.sum(weighted_prob)

    if x is None:
        for i in eval_range:
            final_res[i] = x_prob(i)
        if keep_all:
            return final_res

        final_res_vals = np.array(list(final_res.values()))
        pdf_total = np.sum(final_res_vals)
        if sub is not None:
            for k, v in final_res.items():
                final_res[k] = v / pdf_total
        else:
            if abs(pdf_total - 1.0) > 0.0001:
                raise RuntimeError(
                    "PDF does not sum to 1, got {} which sums to {}.".format(
                        final_res_vals, pdf_total
                    )
                )
        return final_res
    else:
        if sub is not None:
            raise ValueError("x None and sub None together not compatible.")
        return x_prob(x)


def combine_dists(to_eval, dists_dict, dist, sub=None):
    """
    Compute the final weighted distribution for each value in to_eval.

    dist should describe a distribution, which dists_dict depends on
    dists_dict should describe a distribution for each value in dist
    to_eval should be the x-axis in the final distribution

    The function returns the distribution for P(X=x) in to_eval.
    """
    weighted_dist = OrderedDict()
    to_eval, did_sub = subsample_list(to_eval, sub)
    for k in to_eval:
        weighted_dist[k] = 0
    for g in dist.keys():
        full_dist = dists_dict[g]
        prob = dist[g]
        for k in to_eval:
            weighted_dist[k] += prob * full_dist.get(k, 0)

    if did_sub:
        weighted_dist = dist_from_samples(weighted_dist)

    pdf_total = np.sum(np.array(list(weighted_dist.values())))
    if abs(pdf_total - 1.0) > 0.0001:
        raise RuntimeError("PDF does not sum to 1, got {}.".format(pdf_total))

    return weighted_dist


def create_normal(to_eval, mean, var, sub=None, interp_after=False):
    """Create a normal distribution, discretised to a range."""
    final_res = OrderedDict()
    if var == 0:
        final_res[int(mean)] = 1.0
        return final_res

    normal = stats.norm(loc=mean, scale=sqrt(var))
    to_eval_t, did_sub = subsample_list(to_eval, sub)
    vals = normal.pdf(to_eval_t)

    if interp_after and did_sub:
        x_samps = list(to_eval)
        xvals = to_eval_t
        yvals = vals

        vals = interp(x_samps, xvals, yvals)

    else:
        to_eval = to_eval_t

    for e, val in zip(to_eval, vals):
        final_res[int(e)] = val

    return final_res


def get_uniform_moments(min_val, max_val):
    """Get mean, var of a uniform distribution."""
    mean = (max_val + min_val) / 2
    var = (1 / 12) * (max_val - min_val) * (max_val - min_val)

    return mean, var


def subsample_list(list_to_sub, sub_rate):
    """Subsample a list that is assumed to have linearly spaced values."""
    to_eval_temp = list(list_to_sub)
    if sub_rate is not None:
        min_ev = min(to_eval_temp)
        max_ev = max(to_eval_temp)
        step = int(floor(max_ev - min_ev) * sub_rate)
        if step == 0:
            return to_eval_temp, False
        else:
            to_eval = list(range(min_ev, max_ev, step))
        if max_ev not in to_eval:
            to_eval.append(max_ev)
        return to_eval, True
    else:
        return to_eval_temp, False


def dist_from_file(filename):
    """Load a distribution from a file."""
    with open(filename, "r") as f:
        od = OrderedDict()
        lines = f.readlines()
        for line in lines:
            pieces = line.split(":")
            k = pieces[0].strip()
            v = float(pieces[1].strip())
            od[k] = v
    summed = np.sum(np.array(list(od.values())))
    if abs(summed - 1.0) > 0.00001:
        raise RuntimeError("Distribution does not sum to 1, got {}".format(summed))
    return od


def get_dist_ci(dist):
    """Return 95% confidence interval."""
    mean = get_dist_mean(dist)
    var = get_dist_var(dist)

    lower = mean - 1.96 * sqrt(var)
    upper = mean + 1.96 * sqrt(var)

    return lower, upper


def get_dist_ci_alt(dist, ci=False):
    val = 0
    lower = "NA"
    upper = "NA"
    min_k = min(sorted(list(dist.keys())[1:]))
    for key, value in dist.items():
        val = val + (100 * value)
        if (val >= 2.5) and (lower == "NA"):
            lower = key
        if val >= 97.5 and (upper == "NA"):
            upper = key

    if ci:
        return lower - min_k, upper - min_k
    else:
        return lower, upper


def sample_from_dist(dist, n_samples):
    """
    Get samples from a distribution

    Parameters
    ----------
    dist : OrderedDict
        The distribution to sample from
    n_samples : int
        The number of samples to pull from the distribution.

    """
    values = list(dist.keys())
    probabilities = list(dist.values())

    return np.random.choice(values, n_samples, replace=True, p=probabilities)


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
