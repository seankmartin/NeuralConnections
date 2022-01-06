"""Connection stats using the mpmath package."""
import mpmath
from collections import OrderedDict

from .connect_math import hypergeometric_pmf, get_dist_mean, marginal_prob


class CombProb:
    """
    Handle computations with brain region connectivity probabilities.

    Parameters
    ----------
    neurons_A : int
        The number of neurons in brain region A.
    recorded_A : int
        The number of neurons recorded in A.
    connected_A : int
        The number of neurons in A that are
        directly connected to neurons in B.
    neurons_B : int
        The number of neurons in brain region B.
    recorded_B : int
        The number of neurons recorded in B.
    delta_fn : function
        A function of the number of recorded units in A
        which are connected to neurons in B.
        Represents the number of neurons in B which
        receive connections from the recorded neurons in A.
        Must return a distribution over the number of neurons,
        represented as an ordered dictionary.
    delta_params : dict
        keyword arguments to pass into delta_fn

    """

    def __init__(
        self,
        neurons_A,
        recorded_A,
        connected_A,
        neurons_B,
        recorded_B,
        delta_fn,
        cache=True,
        subsample_rate=None,
        approx_hypergeo=False,
        **delta_params
    ):
        """Please see help(CombProb) for more info."""
        self.N = mpmath.mpf(neurons_A)
        self.n = mpmath.mpf(recorded_A)
        self.K = mpmath.mpf(connected_A)
        self.M = mpmath.mpf(neurons_B)
        self.m = mpmath.mpf(recorded_B)
        self.delta_fn = delta_fn
        self.delta_params = delta_params
        self.delta_params["max"] = self.delta_params.get("max", self.M)
        self.verbose = False
        self.cache = cache
        self.stored = OrderedDict()
        self.each_dist = None
        self.a_to_b_dist = None
        if (self.delta_params["max_depth"] > 1) and self.delta_params.get(
            "use_mean", True
        ):
            self.subsample_rate = None
        else:
            self.subsample_rate = subsample_rate
        self.delta_params["subsample_rate"] = self.subsample_rate
        self.approx_hypergeo = approx_hypergeo

        # Use these to check marginal accuracy
        self.plot = False
        self.true_plot = False

        if self.cache:
            self.each_dist, self.a_to_b_dist = self.delta_fn(**delta_params)
            self.stored = self.final_distribution()

            total = mpmath.nsum(self._prob, [0, self.m], verbose=self.verbose)
            if not mpmath.almosteq(total, 1.0, rel_eps=0.001):
                raise RuntimeError(
                    "Total probability does not sum to 1, got {}".format(total)
                )

    def calculate_distribution_n_senders(self):
        """
        Use the hypergeometric distribution to calculate distribution of senders.

        Parameters
        ----------
        None

        Returns
        -------
        OrderedDict
            The full distribution of the number of senders sampled.

        """
        prob_a_senders = OrderedDict()
        for i in range(int(self.n) + 1):
            prob_a_senders[i] = float(hypergeometric_pmf(self.N, self.K, self.n, i))
        return prob_a_senders

    def final_distribution(self, x=None, keep_all=False):
        """
        Calculate P(X=x) as sum P(A_B = k) * P(X=x|k).

        Parameters
        ----------
        x : int, optional
            If provided, calculates only that probability.

        Returns
        -------
        OrderedDict or float
            The full distribution or a single value, depending on x.

        """

        def fn_to_eval(k, i):
            return float(
                hypergeometric_pmf(self.M, k, self.m, i, approx=self.approx_hypergeo)
            )

        non_zeros = 0
        if self.subsample_rate is not None:
            for k, v in self.a_to_b_dist.items():
                if v != 0.0:
                    non_zeros += 1
            if non_zeros > self.subsample_rate * len(list(self.a_to_b_dist.keys())):
                sub = self.subsample_rate
            else:
                sub = None
        else:
            sub = None

        estimated = marginal_prob(
            range(int(self.m) + 1),
            self.a_to_b_dist,
            fn_to_eval,
            x=x,
            sub=sub,
            plot=self.plot,
            true_plot=self.true_plot,
            keep_all=keep_all,
        )
        return estimated

    def connection_prob(self, x):
        """
        Calculate the probability of observing x
        connections between brain regions A and B.

        Parameters
        ----------
        x : int
            The number of connections between A and B.

        Returns
        -------
        mpmath.mpf
            The probability of observing x connections.
        """
        return self.final_distribution(x)

    def geq_connection_prob(self, x):
        """
        Calculate the probability of observing >=x
        connections between brain regions A and B.

        Parameters
        ----------
        x : int
            The min number of connections between A and B.

        Returns
        -------
        mpmath.mpf
            The probability of observing >=x connections.
        """
        return 1 - mpmath.nsum(self.connection_prob, [0, x - 1], verbose=self.verbose)

    def expected_connections(self):
        """
        Calculate expected number of connections between A and B.

        Returns
        -------
        mpmath.mpf
            The expected number of connections between A and B.

        """
        return mpmath.nsum(self._weighted_val, [0, self.m], verbose=self.verbose)

    def run(self, x):
        """
        Helper function to run useful methods for given x.

        Parameters
        ----------
        x : int
            The number of connections between A and B.

        Returns
        -------
        Dict
            keys are:
            "prob" - connection_prob(x)
            "prob_geq" - geq_connection_prob(x)
            "expected" - expected_connections()

        """
        out_dict = {}
        out_dict["prob"] = self.connection_prob(x)
        out_dict["prob_geq"] = self.geq_connection_prob(x)
        out_dict["expected"] = self.expected_connections()
        return out_dict

    def set_verbose(self, vb):
        """Set whether to print more information."""
        self.verbose = vb

    def get_all_prob(self):
        """Return cached probabilities to speed up operations."""
        if not self.cache:
            raise ValueError("Set cache to true to store all probabilities")
        return self.stored

    def expected_total(self, k):
        """Return the expected total in B with k sender samples."""
        return get_dist_mean(self.each_dist[k])

    def _weighted_val(self, x):
        return mpmath.mpf(x) * self._get_val(x)

    def _prob(self, x):
        return self._get_val(x)

    def _get_val(self, x):
        if self.cache:
            if int(x) not in self.stored:
                res = self.connection_prob(x)
                self.stored[int(x)] = res
            else:
                res = self.stored[int(x)]
        else:
            res = self.connection_prob(x)
        return res


if __name__ == "__main__":
    from connect_math import expected_unique

    # def delta_fn(k, **delta_params):
    #     max_val = delta_params.get("max", 10000)
    #     return min(mpmath.mpf(k) * 10, mpmath.mpf(max_val))
    # def delta_fn(k, **delta_params):
    #     N = delta_params.get("N")
    #     K = delta_params.get("K")
    #     n = delta_params.get("n")
    #     exp = 0
    #     for i in range(n):
    #         exp += i * hypergeometric_pmf(10, 2, k, i)
    #     print(k, (2 * k) - exp)
    #     return (2 * k) - exp

    def delta_fn(k, **delta_params):
        N = delta_params.get("N")
        connections = delta_params.get("connections")
        return expected_unique(N, k * connections)

    cp = CombProb(10, 2, 10, 10, 2, delta_fn, N=10, connections=2)
    # cp = CombProb(
    #     10000, 10, 1000, 10000, 10, delta_fn,
    #     N=10000, connections=100)
    cp.set_verbose(True)
    print(cp.run(1))
    # print(cp.run(100))
