# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from ..sub_terrain_cfg import SubTerrainBaseCfg

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshPlaneTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.flat_terrain


@configclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = mesh_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float = MISSING
    """The width of the steps (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    platform_height: float = -1.0
    """The height of the platform.  Defaults to -1.0.

    If the value is negative, the height is the same as the object height."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """


@configclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = mesh_terrains.inverted_pyramid_stairs_terrain


@configclass
class MeshRandomGridTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a random grid mesh terrain."""

    function = mesh_terrains.random_grid_terrain

    grid_width: float = MISSING
    """The width of the grid cells (in m)."""
    grid_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the grid cells (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have randomized grid cells only along the plane extending
    from the platform (like a plus sign). The remaining area remains empty and no border will be added.
    """


@configclass
class MeshRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""

    function = mesh_terrains.rails_terrain

    rail_thickness_range: tuple[float, float] = MISSING
    """The thickness of the inner and outer rails (in m)."""
    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshPitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a pit that leads out of the pit."""

    function = mesh_terrains.pit_terrain

    pit_depth_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the pit (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_pit: bool = False
    """If True, the pit contains two levels of stairs. Defaults to False."""


@configclass
class MeshBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with boxes (similar to a pyramid)."""

    function = mesh_terrains.box_terrain

    box_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""
    double_box: bool = False
    """If True, the pit contains two levels of stairs/boxes. Defaults to False."""


@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""

    function = mesh_terrains.gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gap (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshGapStripTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a repeated gap-and-landing strip along +X."""

    function = mesh_terrains.gap_strip_terrain

    gap_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the gaps (in m)."""
    landing_length: float = 0.5
    """Length of each landing platform between gaps (in m)."""
    start_platform_length: float = 3.0
    """Length of the initial run-up platform (in m)."""
    platform_width: float = 3.0
    """Alias for the run-up platform length for spawn clamping."""


@configclass
class MeshHurdleStripTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a repeated hurdle strip along +X with a run-up platform."""

    function = mesh_terrains.hurdle_strip_terrain

    hurdle_height_range: tuple[float, float] = (0.05, 0.3)
    """Min/max hurdle height (in m) over difficulty."""
    hurdle_thickness: float = 0.2
    """Thickness of each hurdle block along X (in m)."""
    hurdle_gap_range: tuple[float, float] = (0.7, 2.0)
    """Min/max flat gap between hurdles (in m) over difficulty."""
    start_platform_length: float = 3.0
    """Run-up length before the first hurdle (in m)."""


@configclass
class MeshStairsStripTerrainCfg(SubTerrainBaseCfg):
    """Configuration for repeated up/down stair segments along +X with a run-up."""

    function = mesh_terrains.stairs_strip_terrain

    start_platform_length: float = 3.0
    """Run-up length before the first stair (in m)."""
    segment_length: float = 4.0
    """Length of each stair segment (in m)."""
    step_height_range: tuple[float, float] = (0.05, 0.23)
    """Min/max total height change per segment (in m) over difficulty."""
    steps_per_segment: int = 10
    """Number of steps per segment."""
    pattern: tuple[str, ...] = ("up", "down", "up", "down")
    """Sequence of segments; each element is 'up' or 'down'."""


@configclass
class MeshParkourStepTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a parkour-style staircase that rises then descends."""

    function = mesh_terrains.parkour_step_terrain

    start_platform_length: float = 3.0
    """Run-up length before the first step (in m)."""
    step_height_range: tuple[float, float] = (0.1, 0.45)
    """Min/max height change per step (in m) over difficulty."""
    step_length_base_range: tuple[float, float] = (0.3, 1.5)
    """Base step length range; actual length = base + step_height (in m)."""
    steps: int = 6
    """Total number of steps (rise then fall)."""


@configclass
class MeshFloatingRingTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a floating ring around the center."""

    function = mesh_terrains.floating_ring_terrain

    ring_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the ring (in m)."""
    ring_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the ring (in m)."""
    ring_thickness: float = MISSING
    """The thickness (along z) of the ring (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshStarTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a star pattern."""

    function = mesh_terrains.star_terrain

    num_bars: int = MISSING
    """The number of bars per-side the star. Must be greater than 2."""
    bar_width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the bars in the star (in m)."""
    bar_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the bars in the star (in m)."""
    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""


@configclass
class MeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        height: float = MISSING
        """The height (along z) of the object (in m)."""

    function = mesh_terrains.repeated_objects_terrain

    object_type: Literal["cylinder", "box", "cone"] | callable = MISSING
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """
    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""

    max_height_noise: float | None = None
    """"This parameter is deprecated, but stated here to support backward compatibility"""

    abs_height_noise: tuple[float, float] = (0.0, 0.0)
    """The minimum and maximum amount of additive noise for the height of the objects. Default is set to 0.0, which is no noise."""

    rel_height_noise: tuple[float, float] = (1.0, 1.0)
    """The minimum and maximum amount of multiplicative noise for the height of the objects. Default is set to 1.0, which is no noise."""

    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""

    def __post_init__(self):
        if self.max_height_noise is not None:
            warnings.warn(
                "MeshRepeatedObjectsTerrainCfg: max_height_noise:float is deprecated and support will be removed in the"
                " future. Use abs_height_noise:list[float] instead."
            )
            self.abs_height_noise = (-self.max_height_noise, self.max_height_noise)


@configclass
class MeshDebrisTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a debris field with mixed boxes and cylinders."""

    function = mesh_terrains.debris_terrain

    num_debris_min: int = 20
    """Minimum number of debris primitives (at difficulty=0)."""
    num_debris_max: int = 40
    """Maximum number of debris primitives (at difficulty=1)."""
    ground_thickness: float = 0.1
    """Thickness of the base ground slab (in m)."""
    box_length_range: tuple[float, float] = (0.5, 2.0)
    """Range for box length (x) in meters (laid flat)."""
    box_width_range: tuple[float, float] = (0.2, 0.6)
    """Range for box width (y) in meters (laid flat)."""
    box_thickness_range: tuple[float, float] = (0.05, 0.25)
    """Range for box thickness (z) in meters. Samples grow with difficulty."""
    cyl_radius_range: tuple[float, float] = (0.05, 0.25)
    """Range for cylinder radius in meters. Samples grow with difficulty."""
    cyl_length_range: tuple[float, float] = (0.5, 2.0)
    """Range for cylinder length in meters (laid horizontally)."""
    seed: int | None = None
    """Optional seed for deterministic debris placement."""


@configclass
class MeshRepeatedPyramidsTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated pyramids."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for a curriculum of repeated pyramids."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cone

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""


@configclass
class MeshRepeatedBoxesTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated boxes."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated boxes."""

        size: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_box

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""


@configclass
class MeshRepeatedCylindersTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated cylinders."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated cylinder."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cylinder

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""
