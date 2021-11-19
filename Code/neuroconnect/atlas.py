"""Functions related to atlas processing"""

from pprint import pprint

import vedo
import numpy as np
import matplotlib.pyplot as plt
from bg_atlasapi import BrainGlobeAtlas, show_atlases
from brainrender import Scene
from brainrender.actors import Points
from skm_pyutils.py_plot import ColorManager


def vedo_vis(regions, colors=None, atlas_name="allen_mouse_25um"):
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

def brainrender_vis(regions, colors=None, atlas_name="allen_mouse_25um"):

    if colors is None:
        cm = ColorManager(num_colors=len(regions), method="rgb")
        colors = cm.colors

    def get_n_random_points_in_region(region, N):
        """
        Gets N random points inside (or on the surface) of a mes
        """

        region_bounds = region.mesh.bounds()
        print(region_bounds)
        X = np.random.randint(region_bounds[0], region_bounds[1], size=10000)
        Y = np.random.randint(region_bounds[2], region_bounds[3], size=10000)
        Z = np.random.randint(region_bounds[4], region_bounds[5], size=10000)
        pts = [[x, y, z] for x, y, z in zip(X, Y, Z)]

        ipts = region.mesh.insidePoints(pts).points()
        
        if N < ipts.shape[0]:
            return ipts[np.random.choice(ipts.shape[0], N, replace=False), :]
        else:
            return ipts


    scene = Scene(title="Labelled cells")

    # Get a numpy array with (fake) coordinates of some labelled cells
    for region, color in zip(regions, colors):
        print(color)
        mos = scene.add_brain_region(region, alpha=0.15)
        coordinates = get_n_random_points_in_region(mos, 2000)
        color = [color] * coordinates.shape[0]

        # Add to scene
        scene.add(Points(coordinates, name=f"{region} CELLS", colors=color))

    # render
    scene.content
    scene.render()


if __name__ == "__main__":
    show_atlases()
    main_regions = ["CA", "PL"]
    # main_colours = ["k", "b"]
    vedo_vis(main_regions, None)
    brainrender_vis(main_regions, None)
