# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed Pose IK Action for testing robot stability.

This action ignores input commands and always targets a fixed EE pose
captured at reset. Use this to test if the robot can hold a static pose
without drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


class FixedPoseIKAction(DifferentialInverseKinematicsAction):
    """IK action that holds a fixed pose captured at reset.
    
    This action term ignores the input actions and always sets the IK target
    to a fixed EE pose that was captured at reset. This is useful for testing
    whether the robot can maintain a static pose without drift.
    """
    
    def __init__(self, cfg: "FixedPoseIKActionCfg", env):
        super().__init__(cfg, env)
        # Will be populated on first reset
        self._fixed_ee_pos = None
        self._fixed_ee_quat = None
        self._initialized = False
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset and capture current EE pose as fixed target."""
        super().reset(env_ids)
        
        # Compute current EE pose
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        
        # Initialize on first reset
        if not self._initialized:
            self._fixed_ee_pos = ee_pos_curr.clone()
            self._fixed_ee_quat = ee_quat_curr.clone()
            self._initialized = True
            print(f"[FixedPoseIKAction] Captured fixed EE pose:")
            print(f"  pos: {self._fixed_ee_pos[0].tolist()}")
            print(f"  quat: {self._fixed_ee_quat[0].tolist()}")
        else:
            # Update only for reset envs
            if env_ids is not None:
                env_ids_tensor = torch.as_tensor(env_ids, device=self.device)
                self._fixed_ee_pos[env_ids_tensor] = ee_pos_curr[env_ids_tensor]
                self._fixed_ee_quat[env_ids_tensor] = ee_quat_curr[env_ids_tensor]
            else:
                self._fixed_ee_pos[:] = ee_pos_curr
                self._fixed_ee_quat[:] = ee_quat_curr
    
    def process_actions(self, actions: torch.Tensor) -> None:
        """Ignore input actions and force IK target to fixed pose."""
        # Store zero actions (we ignore the input)
        self._raw_actions[:] = 0.0
        self._processed_actions[:] = 0.0
        
        # Force IK target to fixed pose (captured at reset)
        if self._fixed_ee_pos is not None:
            self._ik_controller.ee_pos_des[:] = self._fixed_ee_pos
            self._ik_controller.ee_quat_des[:] = self._fixed_ee_quat


@configclass
class FixedPoseIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for FixedPoseIKAction."""
    
    class_type: type = FixedPoseIKAction
