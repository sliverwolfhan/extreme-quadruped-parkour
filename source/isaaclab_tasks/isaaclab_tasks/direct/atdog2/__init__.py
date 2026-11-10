# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym tasks for ATDog2 on Robocon Meihua Stepping Stones."""

import gymnasium as gym

from . import agents

##
# Register Gym environments for ATDog2.
##

gym.register(
    id="ATDog2-Meihua-Direct-Abl3_5-v0",
    entry_point=f"{__name__}.atdog2_env:ATDog2MeihuaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atdog2_env_cfg:ATDog2MeihuaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDog2MeihuaAbl3_5PPORunnerCfg",
    },
)

gym.register(
    id="ATDog2-Meihua-Direct-Abl3_5-Play-v0",
    entry_point=f"{__name__}.atdog2_env:ATDog2MeihuaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atdog2_env_cfg:ATDog2MeihuaPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDog2MeihuaAbl3_5PPORunnerCfg",
    },
)
