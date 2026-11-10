# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

from .actor_critic_scan import ActorCriticScan  # noqa: F401
import builtins
import rsl_rl.modules
builtins.ActorCriticScan = ActorCriticScan
setattr(rsl_rl.modules, "ActorCriticScan", ActorCriticScan)



@configclass
class RslRlPpoActorCriticScanCfg(RslRlPpoActorCriticCfg):
    """Extended config with scan encoder parameters."""
    num_prop_obs: int | None = None
    num_scan_obs: int | None = None
    num_actor_scan_obs: int | None = None
    num_critic_scan_obs: int | None = None
    scan_encoder_dims: list[int] | None = None
    actor_scan_encoder_dims: list[int] | None = None
    critic_scan_encoder_dims: list[int] | None = None
    encode_scan_for_critic: bool = True
    priv_obs_encoder_dims: list[int] | None = None
    priv_encoder_dims: list[int] | None = None


def _make_atdog2_policy_cfg(
    scan_encoder_dims=None,
    actor_scan_encoder_dims=None,
    critic_scan_encoder_dims=None,
    encode_scan_for_critic: bool = True,
    priv_obs_encoder_dims=None,
    priv_encoder_dims=None,
    num_actor_scan_obs: int | None = None,
    num_critic_scan_obs: int | None = None,
):
    return RslRlPpoActorCriticScanCfg(
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
class ATDog2MeihuaAbl3_5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO Runner configuration for ATDog2 on Robocon Meihua Stepping Stones (Ablation 3.5)."""
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "atdog2_meihua_direct_abl3_5"
    empirical_normalization = True

    policy = _make_atdog2_policy_cfg(scan_encoder_dims=[128, 64, 32])
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
