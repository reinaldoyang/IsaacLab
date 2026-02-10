# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

from . import agents

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="TM7S-Plate-Stack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_tm7s_env_cfg:TM7SCubeStackEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="TM7S-Cube-Stack-Mimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_tm7s_env_cfg:TM7SCubeStackEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)