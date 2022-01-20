"""Abstract class setting the functions needed to define connections."""

from abc import ABC, abstractmethod
import math
from collections import OrderedDict

from mpmath import nstr
from scipy import sparse
import numpy as np

from .connect_math import (
    expected_unique,
    expected_non_overlapping,
    expected_overlapping,
    random_draw_dist,
    hypergeometric_pmf,
    get_dist_mean,
    apply_fn_to_dist,
    create_uniform,
    nfold_conv,
    convolution,
    combine_dists,
    create_normal,
    get_uniform_moments,
    get_dist_var,
)
from .simple_graph import from_matrix


def get_by_name(name):
    """Retrieve a connection strategy by name."""
    classes = {
        "recurrent_connectivity": RecurrentConnectivity,
        "matrix_connectivity": MatrixConnectivity,
        "mean_connectivity": MeanRecurrentConnectivity,
        "unique_connectivity": UniqueConnectivity,
    }
    return classes[name]


class ConnectionStrategy(ABC):
    """
    Abstract class to describe the connection strategy between two regions.

    Must define:
    1. create_connections:
        How to form connections between neurons in the regions.
        Expected to return graph, connections
    2. expected_connections:
        How many connections would be expected between neurons in the regions.
    3. static_expected_connections:
        An interface into expected_connections that can be statically called.

    """

    def __init__(self, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def create_connections(self, *args, **kwargs):
        """How to form connections between neurons in the regions."""
        pass

    @abstractmethod
    def expected_connections(self, *args, **kwargs):
        """The distribution of connections."""
        pass

    @staticmethod
    @abstractmethod
    def static_expected_connections(*args, **kwargs):
        """The distribution of connections called statically."""
        pass


class UniqueConnectivity(ConnectionStrategy):
    """
    Random connections where connections don't overlap.

    """

    def __init__(self, **kwargs):
        self.num_senders = kwargs.get("num_senders")
        self.min_forward = kwargs.get("min_forward")
        self.max_forward = kwargs.get("max_forward")

    def create_connections(self, choices, **kwargs):
        """Create connections randomly from model stats."""
        region_verts = kwargs.get("region_verts")
        graph = []

        # Choose the forward connectors
        connected = np.random.choice(region_verts, size=self.num_senders, replace=False)
        forward_connections = np.random.choice(
            choices, size=(self.num_senders, self.max_forward), replace=True
        )
        num_choices = np.random.randint(
            self.min_forward,
            self.max_forward + 1,
            dtype=np.int32,
            size=self.num_senders,
        )

        f_idx = 0
        # Create forward_connections and inter_connections
        for i, vert in enumerate(region_verts):
            self_connections = np.array([], dtype=np.int32)

            # Create forward_connections
            if vert in connected:
                forward_connections = np.random.choice(
                    choices, size=(self.max_forward), replace=False
                )
                forward_connection = forward_connections[: num_choices[f_idx]]
                self_connections = np.append(
                    self_connections, list(set(forward_connection))
                )
                f_idx = f_idx + 1

            if isinstance(self_connections, np.int32):
                graph.append(np.array([self_connections], dtype=np.int32))
            else:
                graph.append(self_connections.astype(np.int32))

        return graph, connected

    def expected_connections(self, num_samples, **kwargs):
        """Calls static_expected_connections"""
        return UniqueConnectivity.static_expected_connections(
            total_samples=num_samples, **kwargs
        )

    @staticmethod
    def static_expected_connections(**kwargs):
        """Return connection distribution."""
        # Parse out the relevant parameters
        max_depth = kwargs.get("max_depth", 1)

        if max_depth > 1:
            raise ValueError("max_depth must be 1 for unique connections.")

        num_end = kwargs.get("N")
        out_connections_dist = kwargs.get("out_connections_dist")
        num_start = kwargs.get("num_start")
        num_senders = kwargs.get("num_senders")

        total_samples = kwargs.get("total_samples")
        clt_start = kwargs.get("clt_start", 30)
        sub = kwargs.get("subsample_rate", 0.01)

        # Setup required regardless of the depth of the connection

        # Gives dist of num outgoing connections from A
        # This tends towards normal distribution by CLT in most cases

        def fn_to_apply(k):
            return int(round(expected_unique(num_end, k)))

        ab_dist = random_draw_dist(
            total_samples,
            out_connections_dist,
            num_end,
            apply_fn=False,
            keep_all=True,
            clt_start=clt_start,
            sub=sub,
        )

        final_dist = ab_dist

        # PMF of num senders sampled
        prob_a_senders = OrderedDict()

        for i in range(total_samples + 1):
            prob_a_senders[i] = float(
                hypergeometric_pmf(num_start, num_senders, total_samples, i)
            )

        weighted_dist = combine_dists(
            range(num_end + 1), final_dist, prob_a_senders, sub=None
        )

        return ab_dist, weighted_dist


class RecurrentConnectivity(ConnectionStrategy):
    """
    Random connection with recursive connections and interconnections allowed.

    In this, a certain number of neurons output a random number of connections
    to completely random neurons with repetition possible.
    This handles both forward and backward connections.
    interconnections are handled by randomly sampling a set of
    neurons for each neuron.
    The rate at which interconnected synapses are formed is kept fixed
    but the number of connections varies as multiple synapses can be formed.

    Should pass the following keyword arguments on initialisation:
    num_senders : int
        The number of neurons which send a connection
    min_inter : float
        The minimum interconnection rate
    max_inter : float
        The maximum interconnection rate
    min_forward : int
        The minimum number of forward connections from one neuron
    max_forward : int
        The maximum number of forward connections from one neuron

    """

    def __init__(self, **kwargs):
        self.num_senders = kwargs.get("num_senders")
        self.min_inter = kwargs.get("min_inter")
        self.max_inter = kwargs.get("max_inter")
        self.min_forward = kwargs.get("min_forward")
        self.max_forward = kwargs.get("max_forward")

    def create_connections(self, choices, **kwargs):
        """Create connections randomly from model stats."""
        region_verts = kwargs.get("region_verts")
        box = kwargs.get("box", False)  # for performance reasons

        graph = []

        # Choose the forward connectors
        connected = np.random.choice(region_verts, size=self.num_senders, replace=False)
        forward_connections = np.random.choice(
            choices, size=(self.num_senders, self.max_forward), replace=True
        )
        num_choices = np.random.randint(
            self.min_forward,
            self.max_forward + 1,
            dtype=np.int32,
            size=self.num_senders,
        )

        f_idx = 0
        # Create connections between neurons in the same region
        if self.max_inter > 0:
            if box:
                num_boxes = 5
                box_like = 4
                box_size = int(math.ceil(len(region_verts) / num_boxes))
                out_box_size = len(region_verts) - box_size

                # Represents in box being box_like times more likely than outside
                # e.g. for 5
                # x = 5y
                # 1 = (1 / num_boxes) * (5 * y) + (1 - 1 / num_boxes) * y
                y_mult = (1 / num_boxes) * (box_like) + (1 - (1 / num_boxes))
                out_box_mult = 1 / y_mult
                in_box_mult = box_like * out_box_mult
                mult = np.array(
                    [
                        (1 / num_boxes) * in_box_mult,
                        (1 - (1 / num_boxes)) * out_box_mult,
                    ]
                )
                rand_amt = self.min_inter + (
                    np.random.rand() * (self.max_inter - self.min_inter)
                )
                sample_sizes = np.ceil(rand_amt * mult * len(region_verts)).astype(int)

                in_box_idx = np.array([i for i in range(box_size)], dtype=np.int32)
                self_connects_box = np.random.choice(
                    in_box_idx,
                    size=(len(region_verts), sample_sizes[0]),
                    replace=True,
                )

                out_box_idx = np.array(
                    [i + box_size for i in range(out_box_size)], dtype=np.int32
                )
                self_connects_outside = np.random.choice(
                    out_box_idx,
                    size=(len(region_verts), sample_sizes[1]),
                    replace=True,
                )

            else:
                self_connects = np.random.choice(
                    region_verts,
                    size=(
                        len(region_verts),
                        int(round(self.max_inter * len(region_verts))),
                    ),
                    replace=True,
                )
                num_choices_inter = np.random.randint(
                    int(round(self.min_inter * len(region_verts))),
                    int(round(self.max_inter * len(region_verts))) + 1,
                    dtype=np.int32,
                    size=len(region_verts),
                )

        # Create forward_connections and inter_connections
        for i, vert in enumerate(region_verts):
            if self.max_inter > 0:
                if box:
                    which_box = int(math.floor(i / box_size))
                    a = int(which_box * box_size)

                    self_connects_box_i = a + self_connects_box[i]

                    self_connects_out_i = self_connects_outside[i]
                    self_connects_out_i[
                        self_connects_out_i < (a + box_size)
                    ] -= box_size

                    self_connections = np.array(
                        list(
                            set(
                                np.concatenate(
                                    (self_connects_box_i, self_connects_out_i)
                                    + np.min(region_verts)
                                )
                            )
                        ),
                        dtype=np.int32,
                    )

                else:
                    self_connections = np.array(
                        list(set(self_connects[i, : num_choices_inter[i]])),
                        dtype=np.int32,
                    )

                # Remove autaptic synapses
                self_connections = np.delete(
                    self_connections, np.where(self_connections == vert)
                )
                for val in self_connections:
                    if isinstance(val, float):
                        print(val, self_connections)
                        exit(-1)

            else:
                self_connections = np.array([], dtype=np.int32)

            # Create forward_connections
            if vert in connected:
                forward_connection = forward_connections[f_idx, : num_choices[f_idx]]
                self_connections = np.append(
                    self_connections, list(set(forward_connection))
                )
                f_idx = f_idx + 1

            if isinstance(self_connections, np.int32):
                graph.append(np.array([self_connections], dtype=np.int32))
            else:
                graph.append(self_connections.astype(np.int32))

        return graph, connected

    def expected_connections(self, num_samples, **kwargs):
        """Calls static_expected_connections"""
        return RecurrentConnectivity.static_expected_connections(
            total_samples=num_samples, **kwargs
        )

    @staticmethod
    def static_expected_connections(**kwargs):
        """Return connection distribution."""
        # Parse out the relevant parameters
        max_depth = kwargs.get("max_depth", 3)

        if max_depth > 3:
            raise ValueError("max_depth must be less than 4 currently.")

        use_mean = kwargs.get("use_mean", True)
        if use_mean and (max_depth > 1):
            return MeanRecurrentConnectivity.static_expected_connections(**kwargs)
        else:
            if max_depth >= 1:
                num_end = kwargs.get("N")
                out_connections_dist = kwargs.get("out_connections_dist")
                num_start = kwargs.get("num_start")
                num_senders = kwargs.get("num_senders")

                total_samples = kwargs.get("total_samples")
                clt_start = kwargs.get("clt_start", 30)
                sub = kwargs.get("subsample_rate", 0.01)

                # Used to specify stats in relation to the recording device(s).
                num_start_probe = kwargs.get("num_start_probe", num_start)
                num_senders_probe = kwargs.get("num_senders_probe", num_senders)
                out_connections_dist_probe = kwargs.get(
                    "out_connections_dist_probe", out_connections_dist
                )
                num_end_probe = kwargs.get("num_end_probe", num_end)

                # Setup required regardless of the depth of the connection
                def fn_to_apply(k):
                    # Ideally, here would use float if large var dist, and int otherwise.
                    # Not a huge difference though
                    # return expected_unique(num_end, k, do_round=False)
                    return expected_unique(num_end_probe, k, do_round=True)

                # Gives dist of num outgoing connections from A
                # This tends towards normal distribution by CLT in most cases
                ab_dist = random_draw_dist(
                    total_samples,
                    out_connections_dist_probe,
                    num_end_probe,
                    apply_fn=False,
                    keep_all=True,
                    clt_start=clt_start,
                    sub=sub,
                )

                dists = OrderedDict()
                for k, v in ab_dist.items():
                    dists[k] = apply_fn_to_dist(v, fn_to_apply, sub=sub)
                ab_un_dist = dists
                final_dist = ab_un_dist

            if max_depth >= 2:
                final_dist = OrderedDict()
                start_inter_dist = kwargs.get("start_inter_dist")
                end_inter_dist = kwargs.get("end_inter_dist")
                start_mean = get_dist_mean(out_connections_dist)
                start_var = get_dist_var(out_connections_dist)

                start_inter_mean = get_dist_mean(start_inter_dist)
                start_inter_var = get_dist_var(start_inter_dist)
                end_inter_mean = get_dist_mean(end_inter_dist)
                end_inter_var = get_dist_var(end_inter_dist)

                ab_cache = OrderedDict()
                ab_cache[0] = OrderedDict()
                ab_cache[0][0] = 1.0
                start_max_val = max(list(out_connections_dist.keys()))
                for i in range(1, num_senders + 1):
                    if i < clt_start:
                        ab_cache[i] = convolution(
                            out_connections_dist, ab_cache[i - 1], sub=sub
                        )
                    else:
                        ab_cache[i] = create_normal(
                            range((start_max_val * i) + 1),
                            start_mean * i,
                            start_var * i,
                            sub=sub,
                            interp_after=True,
                        )

                bb_cache = OrderedDict()
                bb_cache[0] = OrderedDict()
                bb_cache[0][0] = 1.0
                end_inter_max_val = max(list(end_inter_dist.keys()))
                for i in range(1, num_end + 1):
                    if i < clt_start:
                        bb_cache[i] = convolution(
                            end_inter_dist, bb_cache[i - 1], sub=sub
                        )
                    else:
                        bb_cache[i] = create_normal(
                            range((end_inter_max_val * i) + 1),
                            end_inter_mean * i,
                            end_inter_var * i,
                            sub=sub,
                            interp_after=True,
                        )

                # AAB calculation
                if total_samples < clt_start:
                    aa_dist = nfold_conv([start_inter_dist] * total_samples, sub=sub)
                else:
                    max_val = max(list(start_inter_dist.keys()))
                    aa_dist = create_normal(
                        range((max_val * total_samples) + 1),
                        start_inter_mean * total_samples,
                        start_inter_var * total_samples,
                        sub=sub,
                        interp_after=True,
                    )
                aa_sender_dist = OrderedDict()

                aab_dist = OrderedDict()
                abb_dist = OrderedDict()

                for i in range(total_samples + 1):

                    def inside_fn(x):
                        aa_sampled = expected_unique(num_start, x)
                        aa_new = expected_non_overlapping(
                            num_start, total_samples, aa_sampled
                        )
                        aa_senders = expected_overlapping(
                            num_start,
                            num_senders - i,
                            aa_new,
                        )
                        return int(round(aa_senders))

                    aa_sender_dist[i] = apply_fn_to_dist(aa_dist, inside_fn, sub=sub)

                    aab_dist[i] = combine_dists(
                        range((start_max_val * num_senders) + 1),
                        ab_cache,
                        aa_sender_dist[i],
                        sub=None,
                    )
                    abb_dist[i] = combine_dists(
                        range((end_inter_max_val * num_end) + 1),
                        bb_cache,
                        ab_un_dist[i],
                        sub=None,
                    )
                    aab_dist[i] = apply_fn_to_dist(aab_dist[i], fn_to_apply, sub=sub)
                    abb_dist[i] = apply_fn_to_dist(abb_dist[i], fn_to_apply, sub=sub)

                    aab_cache = OrderedDict()
                    abb_cache = OrderedDict()

                    for j in range(num_end + 1):

                        def to_app(x):
                            return min(
                                j + round(expected_non_overlapping(num_end, j, x)),
                                num_end,
                            )

                        if j == num_end:
                            to_add = OrderedDict()
                            to_add[num_end] = 1
                            aab_cache[j] = to_add
                            abb_cache[j] = to_add
                        else:
                            aab_cache[j] = apply_fn_to_dist(
                                aab_dist[i], to_app, sub=sub
                            )
                            abb_cache[j] = apply_fn_to_dist(
                                abb_dist[i], to_app, sub=sub
                            )

                    in_prog = OrderedDict()
                    in_prog = combine_dists(
                        range(num_end + 1), aab_cache, ab_un_dist[i], sub=None
                    )
                    in_prog = combine_dists(
                        range(num_end + 1), abb_cache, in_prog, sub=None
                    )
                    final_dist[i] = in_prog

            if max_depth >= 3:
                recurrent_connections_dist = kwargs.get("recurrent_connections_dist")
                num_recurrent = kwargs.get("num_recurrent")
                end_mean = get_dist_mean(recurrent_connections_dist)
                end_var = get_dist_var(recurrent_connections_dist)

                aaab_dist = OrderedDict()
                aabb_dist = OrderedDict()
                abbb_dist = OrderedDict()
                abab_dist = OrderedDict()

                aaab_cache = OrderedDict()
                aabb_cache = OrderedDict()
                abbb_cache = OrderedDict()
                abab_cache = OrderedDict()

                aa_cache = OrderedDict()
                aa_cache[0] = OrderedDict()
                aa_cache[0][0] = 1.0
                start_inter_max_val = max(list(start_inter_dist.keys()))
                for i in range(1, num_start + 1):
                    if i < clt_start:
                        aa_cache[i] = convolution(
                            start_inter_dist, aa_cache[i - 1], sub=sub
                        )
                    else:
                        aa_cache[i] = create_normal(
                            range((start_inter_max_val * i) + 1),
                            start_inter_mean * i,
                            start_inter_var * i,
                            sub=sub,
                            interp_after=True,
                        )

                ba_cache = OrderedDict()
                ba_cache[0] = OrderedDict()
                ba_cache[0][0] = 1.0
                end_max_val = max(list(recurrent_connections_dist.keys()))
                for i in range(1, num_recurrent + 1):
                    if i < clt_start:
                        ba_cache[i] = convolution(
                            recurrent_connections_dist, ba_cache[i - 1], sub=sub
                        )
                    else:
                        ba_cache[i] = create_normal(
                            range((end_max_val * i) + 1),
                            end_mean * i,
                            end_var * i,
                            sub=sub,
                            interp_after=True,
                        )

                ab_sender_dist = OrderedDict()

                def new_fn(x):
                    return int(round(expected_overlapping(num_end, num_recurrent, x)))

                for k, v in ab_un_dist.items():
                    ab_sender_dist[k] = apply_fn_to_dist(v, new_fn, sub=sub)

                aaa_sender_dist = OrderedDict()

                def new_fn(x):
                    return int(round(expected_unique(num_start, x)))

                aa_un_dist = apply_fn_to_dist(aa_dist, new_fn, sub=sub)
                aaa_dist = combine_dists(
                    range((num_start * start_inter_max_val * start_inter_max_val) + 1),
                    aa_cache,
                    aa_un_dist,
                    sub=None,
                )

                aba_dist = OrderedDict()
                aba_sender_dist = OrderedDict()

                for i in range(total_samples + 1):

                    def inside_fn(x):
                        aaa_sampled = expected_unique(num_start, x)
                        aaa_new = expected_non_overlapping(
                            num_start,
                            total_samples + get_dist_mean(aa_un_dist),
                            aaa_sampled,
                        )
                        aaa_senders = expected_overlapping(
                            num_start,
                            num_senders - get_dist_mean(aa_sender_dist[i]) - i,
                            aaa_new,
                        )
                        return int(round(aaa_senders))

                    def new_fn_k(x):
                        aba_sampled = expected_unique(num_start, x)
                        aba_new = expected_non_overlapping(
                            num_start,
                            total_samples
                            + get_dist_mean(aa_un_dist)
                            + get_dist_mean(aaa_dist),
                            aba_sampled,
                        )
                        aba_senders = expected_overlapping(
                            num_start,
                            num_senders
                            - get_dist_mean(aa_sender_dist[i])
                            - get_dist_mean(aaa_sender_dist[i])
                            - i,
                            aba_new,
                        )
                        return int(round(aba_senders))

                    aaa_sender_dist[i] = apply_fn_to_dist(aaa_dist, inside_fn, sub=sub)

                    aba_dist[i] = combine_dists(
                        range((num_recurrent * end_max_val) + 1),
                        ba_cache,
                        ab_sender_dist[i],
                        sub=None,
                    )
                    aba_sender_dist[i] = apply_fn_to_dist(
                        aba_dist[i], new_fn_k, sub=sub
                    )

                    aaab_dist[i] = combine_dists(
                        range((start_max_val * num_senders) + 1),
                        ab_cache,
                        aaa_sender_dist[i],
                        sub=None,
                    )
                    abab_dist[i] = combine_dists(
                        range((start_max_val * num_senders) + 1),
                        ab_cache,
                        aba_sender_dist[i],
                        sub=None,
                    )
                    aabb_dist[i] = combine_dists(
                        range((end_inter_max_val * num_end) + 1),
                        bb_cache,
                        aab_dist[i],
                        sub=None,
                    )
                    abbb_dist[i] = combine_dists(
                        range((end_inter_max_val * num_end) + 1),
                        bb_cache,
                        abb_dist[i],
                        sub=None,
                    )

                    for j in range(num_end + 1):

                        def to_app(x):
                            return min(
                                j + round(expected_non_overlapping(num_end, j, x)),
                                num_end,
                            )

                        if j == num_end:
                            to_add = OrderedDict()
                            to_add[num_end] = 1
                            aaab_cache[j] = to_add
                            aabb_cache[j] = to_add
                            abab_cache[j] = to_add
                            abbb_cache[j] = to_add
                        else:
                            aaab_cache[j] = apply_fn_to_dist(
                                aaab_dist[i], to_app, sub=sub
                            )
                            aabb_cache[j] = apply_fn_to_dist(
                                aabb_dist[i], to_app, sub=sub
                            )
                            abab_cache[j] = apply_fn_to_dist(
                                abab_dist[i], to_app, sub=sub
                            )
                            abbb_cache[j] = apply_fn_to_dist(
                                abbb_dist[i], to_app, sub=sub
                            )

                    in_prog = OrderedDict()
                    in_prog = combine_dists(
                        range(num_end + 1), aaab_cache, final_dist[i], sub=None
                    )
                    in_prog = combine_dists(
                        range(num_end + 1), aabb_cache, in_prog, sub=None
                    )
                    in_prog = combine_dists(
                        range(num_end + 1), abab_cache, in_prog, sub=None
                    )
                    in_prog = combine_dists(
                        range(num_end + 1), abbb_cache, in_prog, sub=None
                    )
                    final_dist[i] = in_prog

            # PMF of num senders sampled
            prob_a_senders = OrderedDict()

            for i in range(total_samples + 1):
                prob_a_senders[i] = float(
                    hypergeometric_pmf(
                        num_start_probe, num_senders_probe, total_samples, i
                    )
                )

            weighted_dist = combine_dists(
                range(num_end + 1), final_dist, prob_a_senders, sub=None
            )

            return dists, weighted_dist

    @staticmethod
    def nfmt(start, *args):
        start = str(start) + ": ("
        for arg in args:
            start = start + nstr(arg, 5) + ", "
        start = start[:-2] + ")"

        return start


class MeanRecurrentConnectivity(RecurrentConnectivity):
    """
    Similar to RecurrentConnectivity, but uses the mean instead of full dists.

    In this way, it is less accurate than the RecurrentConnectivity, but it will
    be faster to compute, and can be performed without knowledge of the variance
    of the underlying distribution.

    """

    def expected_connections(self, num_samples, **kwargs):
        """Calls static_expected_connections"""
        return MeanRecurrentConnectivity.static_expected_connections(
            num_samples, **kwargs
        )

    @staticmethod
    def static_expected_connections(**kwargs):
        """Distribution of connections from the mean."""
        num_end = kwargs.get("N")
        num_connections = get_dist_mean(kwargs.get("out_connections_dist"))
        num_recurrent_synapses = get_dist_mean(kwargs.get("recurrent_connections_dist"))
        num_start = kwargs.get("num_start")
        num_senders = kwargs.get("num_senders")
        num_recurrent = kwargs.get("num_recurrent")
        inter_connections_start = get_dist_mean(kwargs.get("start_inter_dist"))
        inter_connections_end = get_dist_mean(kwargs.get("end_inter_dist"))
        total_samples = kwargs.get("total_samples")
        max_depth = kwargs.get("max_depth", 3)

        plist = []
        if max_depth > 3:
            raise ValueError("max_depth must be less than 4 currently.")

        dists = OrderedDict()
        for num_sender_samples in range(total_samples + 1):
            final = 0
            if max_depth >= 1:
                # AB
                ab = expected_unique(num_end, num_sender_samples * num_connections)
                plist.append(ab)
                final = final + ab

            if max_depth >= 2:
                # AAB
                aa_sampled = expected_unique(
                    num_start, total_samples * inter_connections_start
                )
                aa_new = expected_non_overlapping(num_start, total_samples, aa_sampled)
                aa_senders = expected_overlapping(
                    num_start,
                    max(num_senders - num_sender_samples, 0),
                    aa_new,
                )
                aab_total = expected_unique(num_end, aa_senders * num_connections)
                aab_less_ab = expected_non_overlapping(num_end, final, aab_total)
                plist.append(aab_total)
                final = final + aab_less_ab

                # ABB
                abb_total = expected_unique(num_end, ab * inter_connections_end)
                abb_less_prev = expected_non_overlapping(num_end, final, abb_total)
                plist.append(abb_total)
                final = final + abb_less_prev

            if max_depth >= 3:
                # AAAB
                aaa_sampled = expected_unique(
                    num_start, aa_sampled * inter_connections_start
                )
                aaa_new = expected_non_overlapping(
                    num_start, aa_new + total_samples, aaa_sampled
                )
                aaa_senders = expected_overlapping(
                    num_start,
                    max(num_senders - (aa_senders + num_sender_samples), 0),
                    aaa_new,
                )
                aaab_total = expected_unique(num_end, aaa_senders * num_connections)
                aaab_less_prev = expected_non_overlapping(num_end, final, aaab_total)
                plist.append(aaab_total)
                final = final + aaab_less_prev

                # AABB
                aabb_total = expected_unique(
                    num_end, aab_less_ab * inter_connections_end
                )
                aabb_less_prev = expected_non_overlapping(num_end, final, aabb_total)
                plist.append(aabb_total)
                final = final + aabb_less_prev

                # ABAB
                ab_recurrent = expected_overlapping(num_end, num_recurrent, ab)
                aba = expected_unique(num_start, ab_recurrent * num_recurrent_synapses)
                aba_new = expected_non_overlapping(
                    num_start, aaa_new + aa_new + total_samples, aba
                )
                aba_send_connections = expected_overlapping(
                    num_start,
                    num_senders - (aaa_senders + aa_senders + num_sender_samples),
                    aba_new,
                )
                abab_total = expected_unique(
                    num_end, aba_send_connections * num_connections
                )
                abab_less_prev = expected_non_overlapping(num_end, final, abab_total)
                plist.append(abab_total)
                final = final + abab_less_prev

                # ABBB
                abb_new = expected_non_overlapping(num_end, aab_less_ab, abb_less_prev)
                abbb_total = expected_unique(num_end, abb_new * inter_connections_end)
                abbb_less_prev = expected_non_overlapping(num_end, final, abbb_total)
                plist.append(abbb_total)
                final = final + abbb_less_prev
            dists[num_sender_samples] = OrderedDict()
            dists[num_sender_samples][int(final)] = 1.0

        # Sums the above distributions to get the marginal
        prob_a_senders = OrderedDict()

        for i in range(total_samples + 1):
            prob_a_senders[i] = float(
                hypergeometric_pmf(num_start, num_senders, total_samples, i)
            )

        weighted_dist = combine_dists(range(num_end + 1), dists, prob_a_senders)

        return dists, weighted_dist


class MatrixConnectivity(ConnectionStrategy):
    """
    Connections from sparse matrices describing the connectivity.

    Attributes
    ----------
    self.ab : scipy.sparse.csr_matrix
        sparse matrix describing forward connections
    self.ba : scipy.sparse.csr_matrix
        sparse matrix describing backward connections
    self.aa : scipy.sparse.csr_matrix
        sparse matrix describing self connections
    self.bb : scipy.sparse.csr_matrix
        sparse matrix describing self connections
    self.to_use : list of bool
        Which matrices to use in the order [ab, ba, aa, bb]
    """

    def __init__(self, **kwargs):
        self.ab = kwargs.get("ab")
        self.ba = kwargs.get("ba")
        self.aa = kwargs.get("aa")
        self.bb = kwargs.get("bb")
        self.to_use = kwargs.get("to_use", [True, True, True, True])

        self.load()

        self.num_a, self.num_b = self.ab.shape
        self.a_indices = np.array([i for i in range(self.num_a)])
        self.b_indices = np.array([i for i in range(self.num_b)])

    def load(self):
        """Load the sparse matrices if they are paths to npz files."""
        if self.to_use[0]:
            if isinstance(self.ab, str):
                self.ab = sparse.load_npz(self.ab)
        if self.to_use[1]:
            if isinstance(self.ba, str):
                self.ba = sparse.load_npz(self.ba)
        if self.to_use[2]:
            if isinstance(self.aa, str):
                self.aa = sparse.load_npz(self.aa)
        if self.to_use[3]:
            if isinstance(self.bb, str):
                self.bb = sparse.load_npz(self.bb)

    def create_connections(self):
        """Create a simple graph representation from the matrices."""
        self.graph = from_matrix(self.ab, self.ba, self.aa, self.bb, self.to_use)

    def compute_stats(self):
        """
        Compute descriptive stats on the matrix connectivity.

        Returns
        -------
        dict
            Descriptive statistics on the matrix connectivity.

        """
        args_dict = {}
        args_dict["N"] = self.num_b
        args_dict["num_start"] = self.num_a
        if self.to_use[0]:
            ab_sum = np.squeeze(np.array(self.ab.sum(axis=1).astype(np.int64)))
            self.num_senders = np.count_nonzero(ab_sum)
            if self.num_senders != 0:
                args_dict["num_senders"] = self.num_senders
                self.num_connections = OrderedDict()
                dist = np.bincount(ab_sum)
                for i in range(
                    int(np.amin(ab_sum[ab_sum != 0])), int(np.amax(ab_sum) + 1)
                ):
                    self.num_connections[i] = dist[i] / float(self.num_senders)
                args_dict["out_connections_dist"] = self.num_connections
        if self.to_use[1]:
            ba_sum = np.squeeze(np.array(self.ba.sum(axis=1).astype(np.int64)))
            self.num_recurrent = np.count_nonzero(ba_sum)
            args_dict["num_recurrent"] = self.num_recurrent
            if self.num_recurrent != 0:
                dist = np.bincount(ba_sum)
                self.num_recurrent_connections = OrderedDict()
                for i in range(
                    int(np.amin(ba_sum[ba_sum != 0])), int(np.amax(ba_sum) + 1)
                ):
                    self.num_recurrent_connections[i] = dist[i] / float(
                        self.num_recurrent
                    )
                args_dict["recurrent_connections_dist"] = self.num_recurrent_connections
        if self.to_use[2]:
            aa_sum = np.squeeze(np.array(self.aa.sum(axis=1).astype(np.int64)))
            dist = np.bincount(aa_sum)
            total = aa_sum.shape[0]
            self.inter_connections_start = OrderedDict()
            for i in range(int(np.amin(aa_sum)), int(np.amax(aa_sum) + 1)):
                self.inter_connections_start[i] = dist[i] / float(total)
            args_dict["start_inter_dist"] = self.inter_connections_start
        if self.to_use[3]:
            bb_sum = np.squeeze(np.array(self.bb.sum(axis=1).astype(np.int64)))
            dist = np.bincount(bb_sum)
            total = bb_sum.shape[0]
            self.inter_connections_end = OrderedDict()
            for i in range(int(np.amin(bb_sum)), int(np.amax(bb_sum) + 1)):
                self.inter_connections_end[i] = dist[i] / float(total)
            args_dict["end_inter_dist"] = self.inter_connections_end

        return args_dict

    def compute_probe_stats(self, A_idx, B_idx):
        """
        Compute descriptive stats on the matrix connectivity relative to device.

        Parameters
        ----------
        A_idx : list of int
            The list of integers inside the recording device in region A.
        B_idx : list of int
            The list of integers inside the recording device in region B.

        Returns
        -------
        MatrixConnectivity
            The subsampled connectivity matrix.
        dict
            Descriptive statistics on the matrix connectivity relative to device.

        """
        subsampled_to_probes = self.subsample(A_idx, B_idx)
        return subsampled_to_probes, subsampled_to_probes.compute_stats()

    def expected_connections(self, total_samples, max_depth):
        """Call to static_expected_connections."""
        args_dict = self.compute_stats()
        args_dict["total_samples"] = total_samples
        args_dict["max_depth"] = max_depth
        return MatrixConnectivity.static_expected_connections(**args_dict)

    @staticmethod
    def static_expected_connections(**kwargs):
        """Calls RecurrentConnectivity with the matrix stats."""
        if kwargs.get("mean_estimate", False) is True:
            return MeanRecurrentConnectivity.static_expected_connections(**kwargs)
        else:
            return RecurrentConnectivity.static_expected_connections(**kwargs)

    def subsample(self, num_a, num_b):
        """
        Subsample the connectivity matrices.

        Parameters
        ----------
        num_a : int or list
            The number of indices to randomly select
            OR the actual indices
            must be the same type as num_b
        num_b : int or list
            The number of indices to randomly select
            OR the actual indices
            must be the same type as num_a

        Returns
        -------
        MatrixConnectivity
            The subsampled connectivity matrix.

        """
        if type(num_a) is int:
            a_samples, b_samples = self.gen_random_samples((num_a, num_b))
        else:
            a_samples, b_samples = num_a, num_b

        if self.to_use[0]:
            if a_samples is None:
                ab = self.ab[:, b_samples]
            elif b_samples is None:
                ab = self.ab[a_samples, :]
            else:
                ab_grid = np.ix_(a_samples, b_samples)
                ab = self.ab[ab_grid]
        if self.to_use[1]:
            if a_samples is None:
                ba = self.ba[b_samples, :]
            elif b_samples is None:
                ba = self.ba[:, a_samples]
            else:
                ba_grid = np.ix_(b_samples, a_samples)
                ba = self.ba[ba_grid]
        if self.to_use[2]:
            if a_samples is not None:
                aa_grid = np.ix_(a_samples, a_samples)
                aa = self.aa[aa_grid]
            else:
                aa = self.aa
        if self.to_use[3]:
            if b_samples is not None:
                bb_grid = np.ix_(b_samples, b_samples)
                bb = self.bb[bb_grid]
            else:
                bb = self.bb

        new_mc = MatrixConnectivity(
            aa=aa,
            bb=bb,
            ab=ab,
            ba=ba,
            to_use=self.to_use,
        )

        return new_mc

    def gen_random_samples(self, num_sampled, zeroed=True):
        """Generate random sample indices from both regions."""
        start = np.random.choice(self.a_indices, size=num_sampled[0], replace=False)
        end = np.random.choice(self.b_indices, size=num_sampled[1], replace=False)
        if not zeroed:
            end = end + self.num_a

        return start, end

    def __str__(self):
        return f"AA: {self.aa.shape}, BB: {self.bb.shape}, AB: {self.ab.shape}, BA: {self.ba.shape}"
