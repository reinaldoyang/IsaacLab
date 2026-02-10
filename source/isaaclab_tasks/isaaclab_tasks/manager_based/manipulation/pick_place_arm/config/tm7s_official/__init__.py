# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# TM7S Base Environment (Robot + Table only)
##

gym.register(
    id="TM7S-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tm7s_base_env_cfg:TM7SBaseEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="TM7S-Base-IK-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tm7s_ik_rel_env_cfg:TM7SBaseIKRelEnvCfg",
    },
    disable_env_checker=True,
)