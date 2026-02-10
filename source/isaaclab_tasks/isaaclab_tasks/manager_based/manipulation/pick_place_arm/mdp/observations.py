# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos

def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat

def ee_pose_in_base(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """End-effector pose in robot base frame."""
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos_w = ee.data.target_pos_w[:, 0, :]
    ee_quat_w = ee.data.target_quat_w[:, 0, :]

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        base_pos_w, base_quat_w, ee_pos_w, ee_quat_w
    )

    if return_key == "pos":
        return ee_pos_b
    elif return_key == "quat":
        return ee_quat_b
    else:
        return torch.cat((ee_pos_b, ee_quat_b), dim=1)
    
def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Parallel gripper joint positions."""
    robot: Articulation = env.scene[robot_cfg.name]

    if not hasattr(env.cfg, "gripper_joint_names"):
        raise RuntimeError("gripper_joint_names not defined in env cfg")

    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(joint_ids) == 2, "Only parallel grippers supported"

    finger_1 = robot.data.joint_pos[:, joint_ids[0]].unsqueeze(1)
    finger_2 = robot.data.joint_pos[:, joint_ids[1]].unsqueeze(1)

    return torch.cat((finger_1, finger_2), dim=1)

