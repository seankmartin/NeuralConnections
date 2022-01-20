"""Functions to plot graphs."""


def get_positions(region_sizes, as_dict=False):
    """Get linearly spaced positions for plotting region nodes."""
    x = []
    y = []
    for i, r_s in enumerate(region_sizes):
        for j in range(r_s):
            x.append(i)
            y.append(j)
    if as_dict:
        my_dict = {}
        for i, (x_val, y_val) in enumerate(zip(x, y)):
            my_dict[i] = (x_val, y_val)
        return my_dict
    return x, y


def get_colours(graph_size, start_set, end_set, reachable=None):
    """Get colours for nodes."""
    # Setup the colours
    c = []
    if reachable is None:
        reachable = end_set
    for acc_val in range(graph_size):
        if acc_val in start_set:
            c.append("b")
        elif acc_val in end_set:
            if acc_val in reachable:
                c.append("g")
            else:
                c.append("r")
        else:
            c.append("gray")
    return c


def get_colours_extend(graph_size, start_set, end_set, source, target, reachable=None):
    """
    Get colours for nodes including source and target nodes.

    Blue nodes are those in the source set.
    Orange nodes are those in the start set, not in the source set.
    Green nodes are those reachable from the source that are in target.
    Red nodes are those in target that are not reachable from the source.
    All other nodes are grey.

    """
    # Setup the colours
    c = []
    if reachable is None:
        reachable = end_set
    for acc_val in range(graph_size):
        if acc_val in start_set:
            if acc_val in source:
                c.append("dodgerblue")
            else:
                c.append("darkorange")
        elif acc_val in target:
            if acc_val in reachable:
                c.append("g")
            else:
                c.append("r")
        else:
            c.append("gray")
    return c
