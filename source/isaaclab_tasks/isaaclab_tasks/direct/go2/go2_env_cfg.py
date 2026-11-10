# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshDebrisTerrainCfg,
    MeshGapStripTerrainCfg,
    MeshHurdleStripTerrainCfg,
    MeshParkourStepTerrainCfg,
    MeshStairsStripTerrainCfg,
)
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
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
class Go2FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 52
    state_space = 81  # 52 prop + 29 priv
    num_prop_obs: int = 52
    num_priv_obs: int = 29
    num_scan_obs: int = 0
    use_scan_in_policy: bool = False
    use_scan_in_critic: bool = False
    scan_first_in_policy: bool = False
    scan_first_in_critic: bool = False

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
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    # Enable self-collisions and bump solver iterations so limbs don't pass through each other.
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UNITREE_GO2_CFG.spawn.replace(
            articulation_props=UNITREE_GO2_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            )
        ),
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 3.0
    yaw_rate_reward_scale = 1.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 2.0
    undesired_contact_reward_scale = -0.0
    flat_orientation_reward_scale = -0.5
    base_height_reward_scale = -10.0
    torque_reward_scale = 0.0
    stop_penalty_reward_scale = 0.0
    dof_close_to_default_reward_scale = -0.05
    work_reward_scale = -0.01
    hip_pos_reward_scale = -0.5  # penalty on hip joints drifting from defaults
    hip_joint_pattern: str = ".*_hip_joint"
    feet_stumble_reward_scale = -1.0
    feet_stumble_ratio = 4.0  # horizontal force threshold factor vs vertical

    # Training/Playing 모드 전환을 유연하게 하려고 추가한 플래그 
    ## command / curriculum settings (overridden by rough configs as needed)
    command_mode: str = "random"  # "random" or "fixed"
    fixed_command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    use_curriculum: bool = True
    heading_command: bool = False
    heading_resample_time_s: float = 10.0
    heading_control_stiffness: float = 0.5
    rel_heading_envs: float = 1.0
    rel_standing_envs: float = 0.02
    command_heading_range: tuple[float, float] = (-math.pi, math.pi)
    command_yaw_range: tuple[float, float] = (-1.0, 1.0)
    command_log_interval: int = 0
    command_log_env: int = 0


@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
    # gap terrain constants (used for curriculum + spawn offset)
    gap_subterrain_key: str = "gap_bar"
    hurdle_subterrain_key: str = "hurdle_strip"
    gap_spawn_offset: float = -1.0  # spawn further back on gap/hurdle tiles to give more run-up

    # env
    # policy: 52 prop + 187 scan = 239; critic: 52 prop + 29 priv + 187 scan = 268
    observation_space = 239
    state_space = 268
    num_scan_obs: int = 187
    use_scan_in_policy: bool = True
    use_scan_in_critic: bool = True

    sim: SimulationCfg = Go2FlatEnvCfg().sim.replace(
        physx=Go2FlatEnvCfg().sim.physx.replace(gpu_max_rigid_patch_count=12 * 2**15)
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            size=(23.0, 6.0),  # Terrain Size 23m X 6m -> x축으로만 직진하니까!
            num_rows=10,  # level 0~9 단계까지
            num_cols=18,  # gap/hurdle/stairs/parkour each occupy 3 columns, others 2 columns
            sub_terrains={
                # 18컬럼에 7타입 배치 (대체로 균등 비율)
                "boxes": ROUGH_TERRAINS_CFG.sub_terrains["boxes"].replace(
                    proportion=(2 / 18), grid_height_range=(0.025, 0.1)
                ),
                "random_rough": ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].replace(
                    proportion=(2 / 18), noise_range=(0.01, 0.06), noise_step=0.01
                ),
                "debris_field": MeshDebrisTerrainCfg(
                    proportion=(2 / 18),
                    size=(23.0, 23.0),
                    num_debris_min=20,
                    num_debris_max=40,
                    ground_thickness=0.1,
                    box_length_range=(0.5, 2.0),
                    box_width_range=(0.2, 0.6),
                    box_thickness_range=(0.05, 0.25),
                    cyl_radius_range=(0.05, 0.2),
                    cyl_length_range=(0.5, 2.0),
                ),
                "gap_bar": MeshGapStripTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    gap_width_range=(0.1, 0.8),
                    landing_length=0.45,
                    start_platform_length=8.0,  # longer run-up for the gap strip
                ),
                # hurdle strip: run-up then repeated hurdles with height/gap increasing in difficulty
                "hurdle_strip": MeshHurdleStripTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    hurdle_height_range=(0.05, 0.3),
                    hurdle_thickness=0.2,
                    hurdle_gap_range=(0.7, 2.0),
                    start_platform_length=3.0,
                ),
                # stairs strip: run-up then up/down stair segments
                "stairs_strip": MeshStairsStripTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    start_platform_length=3.0,
                    segment_length=5.0,
                    step_height_range=(0.05, 0.23),
                    steps_per_segment=10,
                    pattern=("up", "down", "up", "down"),
                ),
                "parkour_step": MeshParkourStepTerrainCfg(
                    proportion=(3 / 18),
                    size=(23.0, 23.0),
                    start_platform_length=3.0,
                    step_height_range=(0.1, 0.45),
                    step_length_base_range=(0.3, 1.5),
                    steps=6,
                ),
            },
        ),
        max_init_terrain_level=5, # 사수님이 9였다가 1로 바꾸심 -> 내가 5로 바꿈
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Test25 reward scales (override from flat config)
    base_height_reward_scale = 0.0 # Test4 (considering managerbased curriculum)
    flat_orientation_reward_scale = 0.0 # 험지니까 몸이 엄청 기울거라서
    feet_air_time_reward_scale = 0.125 # Test25 (@ Extreme Parkour: 0)
    lin_vel_reward_scale = 3.0 # Test23
    yaw_rate_reward_scale = 1.5 # Test23
    z_vel_reward_scale = -0.0 # Test9 (jumping 많은 지형 고려) # 참고: Extreme Parkour: -0.5
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5 # 참고: Extreme Parkour: -1.0e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01 # Test 22(Hmm didn't overcome Gap so didn't consider "Extreme Parkour: -0.1 scale")
    undesired_contact_reward_scale = -0.8 # Test4 (considering managerbased curriculum)
    torque_reward_scale = 0.0
    stop_penalty_reward_scale = 0.0
    dof_close_to_default_reward_scale = -0.01 # Test24 ((Hmm didn't overcome Gap so didn't consider "Extreme Parkour: -0.04")
    work_reward_scale = -0.003 # Test 12
    hip_pos_reward_scale = -0.3  # Test24 (@ Extreme Parkour: -0.5)
    feet_stumble_reward_scale = -0.0 # Test23 (@ Extreme Parkour: -1.0)
    feet_stumble_ratio = 4.0 # Reward Scale이 아니라, lateral force > fleet_stumble_ratio*|vertical force| 이면 패널티 주도록(옆으로 미끌리는거 방지)

    # keep curriculum active and random commands for training
    command_mode: str = "fixed"
    use_curriculum: bool = True
    heading_command: bool = True
    # heading settings (parkour-like, but keep 10s resample + fixed 1 m/s command)
    command_heading_range: tuple[float, float] = (0.0, 0.0)
    heading_control_stiffness: float = 0.8
    rel_heading_envs: float = 1.0
    rel_standing_envs: float = 0.0
    # keep yaw clamp explicit for clarity
    command_yaw_range: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self):
        self.observation_space = self.num_prop_obs + (self.num_scan_obs if self.use_scan_in_policy else 0)
        self.state_space = self.num_prop_obs + self.num_priv_obs + (self.num_scan_obs if self.use_scan_in_critic else 0)


@configclass
class Go2RoughPlayEnvCfg(Go2RoughEnvCfg):
    """Evaluation configuration for Go2 rough terrain."""

    # disable curriculum updates so difficulty stays fixed per reset
    use_curriculum: bool = False

    # run with fixed velocity commands during playbacks
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # generate a different random rough terrain by changing the seed and disabling curriculum in generator
    terrain = Go2RoughEnvCfg().terrain.replace(
        terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=False, seed=424242)
    )


@configclass
class Go2RoughAbl1EnvCfg(Go2RoughEnvCfg):
    use_scan_in_policy: bool = False


@configclass
class Go2RoughAbl1PlayEnvCfg(Go2RoughPlayEnvCfg):
    use_scan_in_policy: bool = False


@configclass
class Go2RoughAbl2_5EnvCfg(Go2RoughEnvCfg):
    scan_first_in_policy: bool = True
    scan_first_in_critic: bool = True


@configclass
class Go2RoughAbl2_5PlayEnvCfg(Go2RoughPlayEnvCfg):
    scan_first_in_policy: bool = True
    scan_first_in_critic: bool = True
