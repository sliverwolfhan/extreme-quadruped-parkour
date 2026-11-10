# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


def _make_go2_rough_policy_cfg(
    *,
    num_actor_scan_obs: int | None = None,
    num_critic_scan_obs: int | None = None,
    scan_encoder_dims: list[int] | None = None,
    actor_scan_encoder_dims: list[int] | None = None,
    critic_scan_encoder_dims: list[int] | None = None,
    encode_scan_for_critic: bool = True,
    priv_obs_encoder_dims: list[int] | None = None,
    priv_encoder_dims: list[int] | None = None,
) -> RslRlPpoActorCriticCfg:
    if priv_encoder_dims is None:
        priv_encoder_dims = priv_obs_encoder_dims
    return RslRlPpoActorCriticCfg(
        class_name="ActorCriticScan",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_prop_obs=52,
        num_scan_obs=187,
        num_actor_scan_obs=num_actor_scan_obs,
        num_critic_scan_obs=num_critic_scan_obs,
        scan_encoder_dims=scan_encoder_dims,
        actor_scan_encoder_dims=actor_scan_encoder_dims,
        critic_scan_encoder_dims=critic_scan_encoder_dims,
        encode_scan_for_critic=encode_scan_for_critic,
        priv_obs_encoder_dims=priv_obs_encoder_dims,
        priv_encoder_dims=priv_encoder_dims,
    )


@configclass
class Go2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000 # 원래 500이 기본이었는데 너무 짧아서 학습 엉망 go2 철푸덕해버림
    save_interval = 50
    experiment_name = "go2_flat_direct"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticScan",
        init_noise_std= 0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        noise_std_type='log',
        num_prop_obs=52,
        num_scan_obs=0,
        scan_encoder_dims=[128, 64, 32],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000 # 모든 Abl 다 20000 통일
    save_interval = 50
    experiment_name = "go2_rough_direct"
    empirical_normalization = True # Default는 False였는데, 대걸님꺼에 맞춰봄

    # init_noise_std, noise_std_type은 Actor network가 출력한 mean에 더해주는 "std"를 학습할 때 사용하는 변수
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticScan",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_prop_obs=52,
        num_scan_obs=187,
        scan_encoder_dims=[128, 64, 32],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2RoughAbl1PPORunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_rough_direct_abl1"
    policy = _make_go2_rough_policy_cfg(num_actor_scan_obs=0)


@configclass
class Go2RoughAbl2_5PPORunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_rough_direct_abl2_5"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class Go2RoughAbl3_5PPORunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_rough_direct_abl3_5"
    policy = _make_go2_rough_policy_cfg(scan_encoder_dims=[128, 64, 32])


@configclass
class Go2RoughAbl4_0PPORunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_rough_direct_abl4_0"
    policy = _make_go2_rough_policy_cfg(
        scan_encoder_dims=[128, 64, 32],
        encode_scan_for_critic=False,
    )


@configclass
class Go2RoughAbl7_0PPORunnerCfg(Go2RoughPPORunnerCfg):
    experiment_name = "go2_rough_direct_abl7_0"
    policy = _make_go2_rough_policy_cfg(
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
    )
