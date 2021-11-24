"""Functions related to atlas processing"""

from pprint import pprint
import os

import vedo
import brainrender
import numpy as np
import myterial
import matplotlib.pyplot as plt
from bg_atlasapi import BrainGlobeAtlas, show_atlases
from skm_pyutils.py_plot import ColorManager
from one.api import One


def vedo_vis(regions, colors=None, atlas_name="allen_mouse_25um"):
    """Visualise regions of atlas using vedo."""
    bg_atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    pprint(bg_atlas.metadata)

    if colors is None:
        cm = ColorManager(num_colors=len(regions), method="rgb")
        colors = cm.colors

    # Show the full heirarchy
    # pprint(bg_atlas.structures.tree.show())

    # The order of x, y, z in the dataset
    ## This assumes asr layout.
    mapping = [2, 0, 1]
    max_x = bg_atlas.resolution[mapping[0]] * bg_atlas.shape[mapping[0]]
    max_y = bg_atlas.resolution[mapping[1]] * bg_atlas.shape[mapping[1]]
    max_z = bg_atlas.resolution[mapping[2]] * bg_atlas.shape[mapping[2]]
    full_size = bg_atlas.shape[0] * bg_atlas.shape[1] * bg_atlas.shape[2]

    points_list = []
    for region in regions:
        mask = bg_atlas.get_structure_mask(region)
        structure_id = bg_atlas.structures[region]["id"]
        y, z, x = np.nonzero(mask == structure_id)
        vol_pc = round((100 * x.shape[0]) / full_size, 2)
        n_sample_points = x.shape[0] // 500
        choice = np.random.randint(low=0, high=x.shape[0], size=n_sample_points)
        x = x[choice] * bg_atlas.resolution[mapping[0]] - (max_x // 2)
        y = y[choice] * bg_atlas.resolution[mapping[1]] - (max_y // 2)
        z = z[choice] * bg_atlas.resolution[mapping[2]] - (max_z // 2)

        print(f"{region} occupies {vol_pc}% of the full brain")
        points_list.append((x, y, z))

    vedo_points = []
    for val, color in zip(points_list, colors):
        vedo_points.append(vedo.Points(np.array(val)).c(color))
    # axs = vedo.Axes(
    #     vedo_points,
    #     xtitle="X-axis in \mum",
    #     ytitle="Variable Y in \mum",
    #     ztitle="Inverted Z in \mum",
    #     htitle="My \Gamma^2_ijk  plot",
    #     hTitleFont="Kanopus",
    #     hTitleJustify="bottom-right",
    #     hTitleColor="red2",
    #     hTitleSize=0.035,
    #     hTitleOffset=(0, 0.075, 0),
    #     hTitleRotation=45,
    #     zHighlightZero=True,
    #     xyFrameLine=2,
    #     yzFrameLine=1,
    #     zxFrameLine=1,
    #     xyFrameColor="red3",
    #     xyShift=1.05,  # move xy 5% above the top of z-range
    #     yzGrid=True,
    #     zxGrid=True,
    #     zxShift=1.0,
    #     xTitleJustify="bottom-right",
    #     xTitleOffset=-1.175,
    #     xLabelOffset=-1.75,
    #     yLabelRotation=90,
    #     zInverted=True,
    #     tipSize=0.25,
    # )
    axs = vedo.Axes(vedo_points, zInverted=True, zHighlightZero=False)
    to = [0, 0, 0]
    from_ = [7000, 6000, 2000]
    camera = dict(pos=from_, focalPoint=to, viewup=[0, 0, -1])
    vedo.show(*vedo_points, axes=axs, camera=camera).close()


def get_points_in_hemisphere(atlas, region_actor, side="left"):
    """Return all points for a given region on one side"""
    all_points = region_actor.mesh.points()
    points_in_hemisphere = np.array(
        [
            point
            for point in all_points
            if atlas.hemisphere_from_coords(point, as_string=True, microns=True) == side
        ]
    )

    return points_in_hemisphere


def brainrender_vis(regions, colors=None, atlas_name="allen_mouse_25um"):
    """Visualise regions in atlas using brainrender"""

    if colors is None:
        cm = ColorManager(num_colors=len(regions), method="rgb")
        colors = cm.colors

    def get_n_random_points_in_region(region, N):
        """
        Gets N random points inside (or on the surface) of a mes
        """

        region_bounds = region.mesh.bounds()
        X = np.random.randint(region_bounds[0], region_bounds[1], size=10000)
        Y = np.random.randint(region_bounds[2], region_bounds[3], size=10000)
        Z = np.random.randint(region_bounds[4], region_bounds[5], size=10000)
        pts = [[x, y, z] for x, y, z in zip(X, Y, Z)]

        ipts = region.mesh.insidePoints(pts).points()

        if N < ipts.shape[0]:
            return ipts[np.random.choice(ipts.shape[0], N, replace=False), :]
        else:
            return ipts

    scene = brainrender.Scene(root=True, title="Labelled cells", atlas_name=atlas_name)

    # Get a numpy array with (fake) coordinates of some labelled cells
    brain_region_actors = []
    for region, color in zip(regions, colors):
        brain_region = scene.add_brain_region(region, alpha=0.15, color=color)
        coordinates = get_n_random_points_in_region(brain_region, 2000)
        color = [color] * coordinates.shape[0]

        # Add to scene
        scene.add(
            brainrender.actors.Points(coordinates, name=f"{region} CELLS", colors=color)
        )
        brain_region_actors.append(brain_region)

    hemisphere_points = [
        get_points_in_hemisphere(scene.atlas, brain_region_actor)
        for brain_region_actor in brain_region_actors
    ]

    p1 = hemisphere_points[0].mean(axis=0)
    p2 = hemisphere_points[1].mean(axis=0)

    mesh = vedo.shapes.Cylinder(pos=[p1, p2], c="blue", r=100, alpha=0.5)
    cylinder = brainrender.actor.Actor(mesh, name="Cylinder", br_class="Cylinder")

    scene.add(cylinder)

    # render
    scene.content
    scene.render()


def make_probes():
    """Visualise some actual probes that were recorded with."""
    # render a bunch of probes as sets of spheres (one per channel)
    scene = brainrender.Scene()
    scene.root._silhouette_kwargs["lw"] = 1
    scene.root.alpha(0.2)
    probes_locs = load_steinmetz_locations()

    for locs in probes_locs:
        k = int(len(locs) / 374.0)

        for i in range(k):
            points = locs[i * 374 : (i + 1) * 374]
            regs = points.allen_ontology.values

            if "LGd" in regs and ("VISa" in regs or "VISp" in regs):
                color = myterial.salmon_darker
                alpha = 1
                sil = 1
            elif "VISa" in regs:
                color = myterial.salmon_light
                alpha = 1
                sil = 0.5
            else:
                continue

            spheres = brainrender.actors.Points(
                points[["ccf_ap", "ccf_dv", "ccf_lr"]].values,
                colors=color,
                alpha=alpha,
                radius=30,
            )
            spheres = scene.add(spheres)

            p1 = points[["ccf_ap", "ccf_dv", "ccf_lr"]].values[0]
            p2 = points[["ccf_ap", "ccf_dv", "ccf_lr"]].values[-1]
            mesh = vedo.shapes.Cylinder(pos=[p1, p2], c=color, r=100, alpha=0.3)
            cylinder = brainrender.actor.Actor(
                mesh, name="Cylinder", br_class="Cylinder"
            )
            scene.add(cylinder)

            if sil:
                scene.add_silhouette(spheres, lw=sil)

    # Add brain regions
    visp, lgd = scene.add_brain_region(
        "VISp",
        "LGd",
        hemisphere="right",
        alpha=0.3,
        silhouette=False,
        color=myterial.blue_grey_dark,
    )
    visa = scene.add_brain_region(
        "VISa",
        hemisphere="right",
        alpha=0.2,
        silhouette=False,
        color=myterial.blue_grey,
    )
    th = scene.add_brain_region(
        "TH", alpha=0.3, silhouette=False, color=myterial.blue_grey_dark
    )
    th.wireframe()
    scene.add_silhouette(lgd, visp, lw=2)

    camera = {
        "pos": (-16170, -7127, 31776),
        "viewup": (0, -1, 0),
        "clippingRange": (27548, 67414),
        "focalPoint": (7319, 2861, -3942),
        "distance": 43901,
    }

    scene.render(zoom=3.5, camera=camera)
    scene.close()


def steinmetz_brain_regions():
    """Write to file the set of regions in each recording."""
    cache_dir = r"E:\OpenNeuroData\Steinmetz2019\Steinmetz_et_al_2019_9974357\9974357"
    one = One(cache_dir=cache_dir)  # The location of the unarchived data
    sessions = one.search(dataset="trials")

    # Get the location of implanted probes
    brain_regions = []
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "..", "results", "allen_regions.txt"), "w") as f:
        for session in sessions:
            locations = one.load_object(session, "channels", attribute="brainLocation")[
                "brainLocation"
            ]
            brain_regions.append(sorted(list(set(locations["allen_ontology"].values))))
            f.write(str(brain_regions[-1]) + "\n")


def load_steinmetz_locations():
    """
    Load the steinmetz dataset of brain locations from figshare.

    https://figshare.com/articles/dataset/Distributed_coding_of_choice_action_and_engagement_across_the_mouse_brain/9974357
    """
    cache_dir = r"E:\OpenNeuroData\Steinmetz2019\Steinmetz_et_al_2019_9974357\9974357"
    one = One(cache_dir=cache_dir)  # The location of the unarchived data
    sessions = one.search(dataset="trials")
    # session = sessions[0]  # take the first session
    # trials = one.load_object(session, "trials")  # load the trials object
    # print(
    #     trials.intervals
    # )  # trials is a Bunch, values are NumPy arrays or pandas DataFrames
    # print(trials.goCue_times)

    # Get the location of implanted probes
    probes_locs = []
    for session in sessions:
        locations = one.load_object(session, "channels", attribute="brainLocation")[
            "brainLocation"
        ]
        probes_locs.append(locations)

    return probes_locs


def vis_steinmetz_with_regions(region_names, colors=None):
    """Visualise recordings containing regions"""
    probes_locs = load_steinmetz_locations()

    scene = brainrender.Scene()
    scene.root._silhouette_kwargs["lw"] = 1
    scene.root.alpha(0.2)

    if colors is None:
        cm = ColorManager(num_colors=len(region_names) + 2, method="rgb")
        colors = cm.colors

    for locs in probes_locs:
        brain_regions = locs["allen_ontology"].values
        cont = False
        for region_name in region_names:
            if region_name not in brain_regions:
                print(sorted(list(set(brain_regions))))
                cont = True

        if cont:
            continue

        k = int(len(locs) / 374.0)

        for i in range(k):
            points = locs[i * 374 : (i + 1) * 374]
            brain_regions = points["allen_ontology"].values
            cont = True
            for region_name in region_names:
                if region_name in brain_regions:
                    print(sorted(list(set(brain_regions))))
                    cont = False

            if cont:
                continue

            sil = 0.5
            alpha = 0.8
            color_list = [colors[-1]] * len(points)
            spheres = brainrender.actors.Points(
                points[["ccf_ap", "ccf_dv", "ccf_lr"]].values,
                colors=color_list,
                alpha=alpha,
                radius=20,
            )
            spheres = scene.add(spheres)

            p1 = points[["ccf_ap", "ccf_dv", "ccf_lr"]].values[0]
            p2 = points[["ccf_ap", "ccf_dv", "ccf_lr"]].values[-1]
            mesh = vedo.shapes.Cylinder(pos=[p1, p2], c=colors[-2], r=100, alpha=0.3)
            cylinder = brainrender.actor.Actor(
                mesh, name="Cylinder", br_class="Cylinder"
            )
            scene.add(cylinder)

            if sil:
                scene.add_silhouette(spheres, lw=sil)

    # Add brain regions
    for region_name, color in zip(region_names, colors[:-2]):
        reg = scene.add_brain_region(
            region_name, hemisphere="right", alpha=0.3, silhouette=False, color=color
        )
        scene.add_silhouette(reg, lw=2)

    th = scene.add_brain_region(
        "TH", alpha=0.3, silhouette=False, color=myterial.blue_grey_dark
    )
    th.wireframe()

    camera = {
        "pos": (-16170, -7127, 31776),
        "viewup": (0, -1, 0),
        "clippingRange": (27548, 67414),
        "focalPoint": (7319, 2861, -3942),
        "distance": 43901,
    }

    scene.render(zoom=3.5, camera=camera)
    scene.close()


def get_points_in_mesh(points, mesh, N=None):
    ipts = mesh.insidePoints(points).points()
    if N is not None:
        return ipts[np.random.choice(ipts.shape[0], N, replace=False), :]
    else:
        return ipts


if __name__ == "__main__":
    # show_atlases()
    # main_regions = ["CA", "PL"]
    # main_colours = ["k", "b"]
    # vedo_vis(main_regions, None)
    # make_probes()
    steinmetz_brain_regions()
    vis_steinmetz_with_regions(["VISp", "VISl"])

    # main_regions = ["MOp", "SSp-ll"]
    # brainrender_vis(main_regions, None)
