# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2-Direct-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

# Playing Mode: Gym에 새로운 환경 ID인 Go2-Rough-Direct-Play-v0를 등록하는 코드
gym.register(
    id="Go2-Rough-Direct-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl1-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl1PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl2_5-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl2_5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl2_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl3_5-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl3_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl4_0-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl4_0PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl7_0-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl7_0PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl1-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl1PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl2_5-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughAbl2_5PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl2_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl3_5-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl3_5PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl4_0-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl4_0PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Rough-Direct-Abl7_0-Play-v0",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughAbl7_0PPORunnerCfg",
    },
)
