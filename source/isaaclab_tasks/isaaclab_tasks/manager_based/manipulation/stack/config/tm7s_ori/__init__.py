# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

gym.register(
    id="TM7S-Original-Cube-Stack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_ik_rel_tm7s_env_cfg:TM7SOriginalCubeStackEnvCfg",
    },
    disable_env_checker=True,
)

# Simple arm test environment (no gripper)
gym.register(
    id="TM7S-Original-Arm-Test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.arm_test_env_cfg:TM7SOriginalArmTestEnvCfg",
    },
    disable_env_checker=True,
)

