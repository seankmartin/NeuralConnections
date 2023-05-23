from neuroconnect.compound import mouse_region_exp

brain_region_pairs = [
    ["CA1", "SUB"],
    ["CA1", "TH"],
    ["SUB", "TH"],
    ["VISp", "LGv"],
    ["VISp", "TH"],
    ["TH", "LGv"]
]

for n in [10, 15, 20, 25]:
    mouse_region_exp(brain_region_pairs, [1], "mouse_region_exp_{n}.csv", n)