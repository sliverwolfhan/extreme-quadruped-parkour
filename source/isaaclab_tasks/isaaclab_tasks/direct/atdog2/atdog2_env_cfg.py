# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import trimesh

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

try:
    from robot_lab.assets.atdog import AT_DOG2_CFG as ATDOG2_CFG
except ImportError:
    from isaaclab_assets.robots.atdog import ATDOG2_CFG


def meihua_terrain(difficulty: float, cfg: "MeshMeihuaTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate Robocon 2027 Plum Blossom (Meihua) stepping stones terrain along +X."""
    meshes_list: list[trimesh.Trimesh] = []
    ground_thickness = 0.1
    y_center = 0.5 * cfg.size[1]
    post_height = cfg.post_height

    # Curriculum scales block size: difficulty 0.0 -> wide blocks (0.45m), difficulty 1.0 -> 0.20m (Robocon standard)
    diff = float(np.clip(difficulty, 0.0, 1.0))
    block_x = cfg.block_size_min[0] + (1.0 - diff) * (cfg.block_size_max[0] - cfg.block_size_min[0])
    block_y = cfg.block_size_min[1] + (1.0 - diff) * (cfg.block_size_max[1] - cfg.block_size_min[1])

    # 1. Base ground floor under the entire lane (z <= 0)
    base_dim = (cfg.size[0], cfg.size[1], ground_thickness)
    base_center = (0.5 * cfg.size[0], y_center, -0.5 * ground_thickness)
    meshes_list.append(trimesh.creation.box(base_dim, trimesh.transformations.translation_matrix(base_center)))

    # 2. Start platform (robot spawns and accelerates here)
    start_len = cfg.start_platform_length
    start_width = min(cfg.size[1], cfg.platform_width)
    start_dim = (start_len, start_width, post_height)
    start_center = (0.5 * start_len, y_center, 0.5 * post_height)
    meshes_list.append(trimesh.creation.box(start_dim, trimesh.transformations.translation_matrix(start_center)))

    # 3. 6 Meihua discrete stepping posts (Centrosymmetric staggered layout)
    # Matching Robocon 2027: 6 blocks (200x200x200mm), pitch ~0.3m, lateral stagger +/-0.1m
    x_ptr = start_len + cfg.post_start_gap
    stagger_y = cfg.stagger_y_offset
    pitch_x = cfg.post_x_pitch

    for i in range(cfg.num_posts):
        cx = x_ptr + i * pitch_x + 0.5 * block_x
        cy = y_center + (stagger_y if (i % 2 == 0) else -stagger_y)
        cz = 0.5 * post_height
        post_dim = (block_x, block_y, post_height)
        meshes_list.append(trimesh.creation.box(post_dim, trimesh.transformations.translation_matrix((cx, cy, cz))))

    # 4. End landing platform
    x_end_start = x_ptr + cfg.num_posts * pitch_x + cfg.post_start_gap
    end_len = max(2.0, cfg.size[0] - x_end_start)
    end_dim = (end_len, start_width, post_height)
    end_center = (x_end_start + 0.5 * end_len, y_center, 0.5 * post_height)
    meshes_list.append(trimesh.creation.box(end_dim, trimesh.transformations.translation_matrix(end_center)))

    # Spawn origin on start platform at height post_height
    origin = np.array([0.5 * start_len, y_center, post_height])
    return meshes_list, origin


@configclass
class MeshMeihuaTerrainCfg(SubTerrainBaseCfg):
    """Configuration for Robocon 2027 Plum Blossom (Meihua) stepping stones terrain."""
    function = meihua_terrain

    start_platform_length: float = 3.0
    platform_width: float = 1.2
    post_height: float = 0.20
    num_posts: int = 6
    block_size_min: tuple[float, float] = (0.20, 0.20)
    block_size_max: tuple[float, float] = (0.45, 0.45)
    post_x_pitch: float = 0.30
    post_start_gap: float = 0.20
    stagger_y_offset: float = 0.10


@configclass
class EventCfg:
    """Configuration for domain randomization."""
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.8, 1.5),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    randomize_pd = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "distribution": "uniform",
            "operation": "scale",
        },
    )


@configclass
class ATDog2MeihuaEnvCfg(DirectRLEnvCfg):
    """Reinforcement learning environment configuration for ATDog2 on Meihua posts."""

    # episode settings
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    num_prop_obs: int = 52
    num_priv_obs: int = 29
    num_scan_obs: int = 187
    use_scan_in_policy: bool = True
    use_scan_in_critic: bool = True
    scan_first_in_policy: bool = False
    scan_first_in_critic: bool = False

    observation_space: int = 239  # 52 prop + 187 scan
    state_space: int = 268        # 52 prop + 29 priv + 187 scan

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_patch_count=12 * 2**15,
        ),
    )

    # 100% Meihua terrain with 10 curriculum levels
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(14.0, 4.0),
            num_rows=10,  # 10 curriculum difficulty levels (0=easy wide, 9=Robocon exact 200mm)
            num_cols=18,  # 18 columns, all specialized for Meihua posts
            curriculum=True,
            sub_terrains={
                "meihua": MeshMeihuaTerrainCfg(
                    proportion=1.0,
                    size=(14.0, 4.0),
                    start_platform_length=3.0,
                    platform_width=1.2,
                    post_height=0.20,
                    num_posts=6,
                    block_size_min=(0.20, 0.20),
                    block_size_max=(0.45, 0.45),
                    post_x_pitch=0.30,
                    post_start_gap=0.20,
                    stagger_y_offset=0.10,
                ),
            },
        ),
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot: ATDog2
    robot: ArticulationCfg = ATDOG2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # 187D height grid scanner: 1.6m forward x 1.0m lateral, resolution 0.1m
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales
    lin_vel_reward_scale = 3.0
    yaw_rate_reward_scale = 1.5
    z_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -0.2
    base_height_reward_scale = -5.0
    torque_reward_scale = 0.0
    stop_penalty_reward_scale = 0.0
    dof_close_to_default_reward_scale = -0.01
    work_reward_scale = -0.003
    hip_pos_reward_scale = -0.3
    hip_joint_pattern: str = ".*_hip_joint"
    feet_stumble_reward_scale = -0.5
    feet_stumble_ratio = 4.0
    same_post_penalty_scale: float = -2.5

    # commands and curriculum
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (0.8, 0.0, 0.0)  # 0.8 m/s forward speed
    use_curriculum: bool = True
    heading_command: bool = True
    command_heading_range: tuple[float, float] = (0.0, 0.0)
    heading_control_stiffness: float = 0.8
    rel_heading_envs: float = 1.0
    rel_standing_envs: float = 0.0
    command_yaw_range: tuple[float, float] = (-1.0, 1.0)
    command_log_interval: int = 0
    command_log_env: int = 0

    def __post_init__(self):
        self.observation_space = self.num_prop_obs + (self.num_scan_obs if self.use_scan_in_policy else 0)
        self.state_space = self.num_prop_obs + self.num_priv_obs + (self.num_scan_obs if self.use_scan_in_critic else 0)


@configclass
class ATDog2MeihuaPlayEnvCfg(ATDog2MeihuaEnvCfg):
    """Evaluation configuration for ATDog2 on Meihua posts."""
    use_curriculum: bool = False
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (0.8, 0.0, 0.0)

    # Fix curriculum to max difficulty (Level 9) for testing Robocon standard
    terrain = ATDog2MeihuaEnvCfg().terrain.replace(
        max_init_terrain_level=9,
        terrain_generator=ATDog2MeihuaEnvCfg().terrain.terrain_generator.replace(curriculum=False, seed=424242),
    )
