"""Functions related to atlas processing"""

from pprint import pprint
import os

import vedo
import brainrender
import numpy as np
import myterial
from bg_atlasapi import BrainGlobeAtlas, show_atlases
from skm_pyutils.py_plot import ColorManager
from one.api import One
from hilbertcurve.hilbertcurve import HilbertCurve


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
        coordinates = get_n_random_points_in_region(brain_region.mesh, 2000)
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


def steinmetz_brain_regions(cache_dir=None):
    """
    Write to file the set of regions in each recording.

    Parameters
    ----------
    cache_dir : str
        The path to the directory containing the steinmetz dataset.
        By default None, which uses the path on my PC.


    """
    if cache_dir is None:
        cache_dir = (
            r"E:\OpenNeuroData\Steinmetz2019\Steinmetz_et_al_2019_9974357\9974357"
        )
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


def load_steinmetz_locations(cache_dir=None):
    """
    Load the steinmetz dataset of brain locations from figshare.

    https://figshare.com/articles/dataset/Distributed_coding_of_choice_action_and_engagement_across_the_mouse_brain/9974357

    Parameters
    ----------
    cache_dir : str
        The path to the directory containing the steinmetz dataset.
        By default None, which uses the path on my PC.

    Returns
    -------
    probes_locs : dataframe like
        A dataframe like object containing information on the brain location of probes.

    """
    if cache_dir is None:
        cache_dir = (
            r"E:\OpenNeuroData\Steinmetz2019\Steinmetz_et_al_2019_9974357\9974357"
        )
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
    """
    Visualise recordings containing regions

    Parameters
    ----------
    region_names : list of str
        The names of the regions
    colors : list of RGB or str, optional
        The colors to use for visualization, by default None

    Returns
    -------
    None

    """
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


def get_idx_of_points_in_meshes(points, meshes, N=None):
    """
    Find the indices of the points inside the given meshes.

    Parameters
    ----------
    points : list of vedo points
        The points to check.
    meshes : list of vedo meshes
        The meshes to check.
    N : int, optional
        A required number of points to find inside the given meshes, by default None

    Returns
    -------
    list of int
        The indices of the points that are inside the given meshes.

    """
    ipts = [mesh.insidePoints(points, returnIds=True) for mesh in meshes]

    set_of_points = set()
    for pt in ipts:
        for p in pt:
            set_of_points.add(p)
    ipts = list(set_of_points)

    if N is not None:
        return ipts[np.random.choice(ipts.shape[0], N, replace=False), :]
    else:
        return ipts


def get_bounding_probes(region_names, session_id=None):
    """
    Find the probes which intersect given regions and their bounds.

    Parameters
    ----------
    region_names : list of str
        The names of the regions
    session_id : int, optional
        The session to consider, by default None,
        which uses all sessions.

    Returns
    -------
    list of tuples
        each tuple contains
        found_regions, points, mesh
        where found_regions is a list of regions found
        points is a list of probe site locations in AP, DV, LR
        mesh is a vedo mesh bounding the probe

    """
    if session_id is None:
        probes_locs = load_steinmetz_locations()
    else:
        probes_locs = [load_steinmetz_locations()[session_id]]

    info = {}
    for i, locs in enumerate(probes_locs):
        brain_regions = locs["allen_ontology"].values
        cont = False
        for region_name in region_names:
            if region_name not in brain_regions:
                cont = True

        if cont:
            continue

        # Split into single probes per session
        k = int(len(locs) / 374.0)

        info[i] = []
        for j in range(k):
            points = locs[j * 374 : (j + 1) * 374]
            brain_regions = points["allen_ontology"].values
            cont = True
            for region_name in region_names:
                if region_name in brain_regions:
                    found_regions = sorted(list(set(brain_regions)))
                    print(f"Found probe in regions: {found_regions}")
                    cont = False

            if cont:
                continue

            p1 = points[["ccf_ap", "ccf_dv", "ccf_lr"]].values[0]
            p2 = points[["ccf_ap", "ccf_dv", "ccf_lr"]].values[-1]

            # 100micron radius cylinder from top to bottom of probe
            n_pixel_micron_radius = 100
            mesh = vedo.shapes.Cylinder(
                pos=[p1, p2], r=n_pixel_micron_radius, alpha=0.3
            )

            info[i].append([found_regions, points, mesh])

    return info


def get_n_random_points_in_region(region_mesh, N, s=None, sort_=False):
    """
    Gets N random points inside (or on the surface) of a mesh

    If sort_ is True, performs a Hilbert curve sorting process
    """

    region_bounds = region_mesh.bounds()
    if s is None:
        s = int(N * 2)
    X = np.random.randint(region_bounds[0], region_bounds[1], size=s)
    Y = np.random.randint(region_bounds[2], region_bounds[3], size=s)
    Z = np.random.randint(region_bounds[4], region_bounds[5], size=s)
    pts = [[x, y, z] for x, y, z in zip(X, Y, Z)]

    ipts = region_mesh.insidePoints(pts).points()

    if N <= ipts.shape[0]:
        ipts = ipts[np.random.choice(ipts.shape[0], N, replace=False), :]
    else:
        ipts = get_n_random_points_in_region(region_mesh, N, s=int(N * 4))

    if sort_:
        hilbert_dim = int(np.floor(np.log10(np.max(ipts)) / np.log10(2)) + 1)
        hilbert_curve = HilbertCurve(hilbert_dim, 3)
        distances = hilbert_curve.distances_from_points(ipts, match_type=True)
        sorted_idxs = distances.argsort()
        ipts = ipts[sorted_idxs]

    return ipts


def get_brain_region_meshes(region_names, atlas_name, hemisphere="right"):
    # TODO am coverting from brain render
    atlas = brainrender.Atlas(atlas_name=atlas_name)
    root = vedo.load(str(atlas.root_meshfile()))
    atlas.root = root
    # slice to keep only one hemisphere
    if hemisphere == "right":
        plane = atlas.get_plane(pos=root.centerOfMass(), norm=(0, 0, 1))
    elif hemisphere == "left":
        plane = atlas.get_plane(pos=root.centerOfMass(), norm=(0, 0, -1))

    region_meshes = []
    for region_name in region_names:
        region_mesh = vedo.load(str(atlas.meshfile_from_structure(region_name)))
        if hemisphere in ("left", "right"):
            region_mesh.cutWithPlane(
                origin=plane.center, normal=plane.normal,
            )

            region_mesh.cap()
        region_meshes.append(region_mesh)

    return region_meshes


def gen_graph_for_regions(
    region_names,
    region_sizes,
    atlas_name=None,
    session_id=None,
    hemisphere="left",
    sort_=False,
):
    """
    Generate a set of points in 3D space for given regions and intersect with probes.

    Parameters
    ----------
    region_names : list of str
        The names of the regions involved.
    region_sizes : list of int
        The number of cells to place in each brain region respectively.
    atlas_name : str, optional
        The name of the atlas to use, by default None
    session_id : int, optional
        The ID of the recording session to consider, by default None
    hemisphere : str, optional
        The part of the brain to consider.
        "right" or "left" or None, by default "left"
    sort_ : bool, optional
        If True, sort the output cells by a Hilbert curve, by default False

    Returns
    -------
    (region_pts, brain_region_meshes, probes_to_use)
    region_pts : list of tuples
        (cell locations in the probes, indices of the cells inside the probes)
    brain_region_meshes : list of vedo meshes
        The brain region meshes for plotting purposes
    probes_to_use : list of tuple
        The probe information of the probes used

    """
    probe_info = get_bounding_probes(region_names, session_id)
    if len(probe_info) > 1:
        print("Found multiple matching probes for the given brain regions.")
        print("You can visualise these probes using get_bounding_probes method")
        print(f"These were {probe_info}")
        print("Using the first entry for now")

    probes_to_use = list(probe_info.values())[0]
    # TODO could improve efficiency by storing which regions the probes are in
    all_cylinders = [entry[-1] for entry in probes_to_use]
    brain_region_meshes = get_brain_region_meshes(
        region_names, atlas_name, hemisphere=hemisphere
    )

    region_pts = []
    for region_mesh, region_size in zip(brain_region_meshes, region_sizes):
        pts = get_n_random_points_in_region(region_mesh, region_size, sort_=sort_)
        pts_idxs = np.sort(get_idx_of_points_in_meshes(pts, all_cylinders))
        pts = pts[pts_idxs]
        region_pts.append((pts, pts_idxs))

    return region_pts, brain_region_meshes, probes_to_use


def visualise_probe_cells(
    region_names,
    region_sizes,
    atlas_name=None,
    session_id=None,
    hemisphere="left",
    colors=None,
    style="metallic",
):
    """
    Render probes in a recording and the cells in inside probe bounds.

    Parameters
    ----------
    region_names : list of str
        The names of the regions
    region_sizes : list of int
        The number of cells in each region
    atlas_name : str, optional
        The name of the atlas, by default None
    session_id : int, optional
        The ID of the session, by default None
    hemisphere : str, optional
        The side of the brain, by default "left"
    colors : list of str or RGB, optional
        The colors to use, by default None
    style : str, optional
        The style of rendering to use, by default "metallic"

    """
    point_locations, brain_region_meshes, probe_info = gen_graph_for_regions(
        region_names, region_sizes, atlas_name, session_id, hemisphere
    )

    brainrender.settings.SHADER_STYLE = style
    brainrender.settings.SHOW_AXES = False
    scene = brainrender.Scene()

    if colors is None:
        cm = ColorManager(num_colors=len(region_names) + len(probe_info), method="sns")
        colors = cm.colors
    iter_color = iter(colors)

    for name, mesh, points_loc in zip(
        region_names, brain_region_meshes, point_locations
    ):
        points_loc = points_loc[0]
        region_color = next(iter_color)
        brain_mesh = brainrender.actor.Actor(
            mesh, name=name, br_class="brain region", color=region_color, alpha=0.3
        )
        scene.add(brain_mesh)
        scene.add_silhouette(brain_mesh, lw=2)

        color_list = [region_color] * len(points_loc)
        spheres = brainrender.actors.Points(
            points_loc, colors=color_list, alpha=0.5, radius=15,
        )
        spheres = scene.add(spheres)

        scene.add_silhouette(spheres, lw=0.5)

    for probes in probe_info:
        sphere_color = next(iter_color)
        points = probes[1][["ccf_ap", "ccf_dv", "ccf_lr"]].values
        color_list = [sphere_color] * len(points)
        spheres = brainrender.actors.Points(
            points, colors=color_list, alpha=0.6, radius=20,
        )
        spheres = scene.add(spheres)

        cylinder = brainrender.actor.Actor(
            probes[-1],
            name="Cylinder",
            br_class="Cylinder",
            color=sphere_color,
            alpha=0.4,
        )
        scene.add(cylinder)

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


if __name__ == "__main__":
    ### Testing smaller functions
    # show_atlases()
    # main_regions = ["CA", "PL"]
    # main_colours = ["k", "b"]
    # vedo_vis(main_regions, None)
    # make_probes()
    # steinmetz_brain_regions()
    # vis_steinmetz_with_regions(["VISp", "VISl"])
    # main_regions = ["MOp", "SSp-ll"]
    # brainrender_vis(main_regions, None)

    #### Visualise probes and cells
    # for style in ("cartoon", "metallic", "plastic", "shiny", "glossy"):
    style = "cartoon"
    colors = ColorManager(4, method="sns", sns_style="deep").colors
    visualise_probe_cells(
        ["VISp", "VISl"],
        [30000, 10000],
        atlas_name="allen_mouse_25um",
        session_id=None,
        hemisphere="left",
        colors=colors,
        style=style,
    )
