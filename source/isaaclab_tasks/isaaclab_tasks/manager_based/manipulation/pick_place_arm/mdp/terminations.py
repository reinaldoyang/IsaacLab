# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def time_out_only(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Dummy termination function.
    Actual time limit is handled by TerminationTermCfg(time_out=True).
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

def mug_on_plate(
    env: ManagerBasedRLEnv, 
    xy_tol: float = 0.05, 
    z_tol: float = 0.01
) -> torch.Tensor:
    """
    Termination: mug is on top of the plate (AABB overlap in XY, mug above plate in Z).
    """
    # Get mug and plate root positions
    mug_pos = env.scene["mug"].data.root_pos_w  # (num_envs, 3)
    plate_pos = env.scene["plate"].data.root_pos_w  # (num_envs, 3)
    
    # Calculate distances
    xy_dist_x = torch.abs(mug_pos[:, 0] - plate_pos[:, 0])
    xy_dist_y = torch.abs(mug_pos[:, 1] - plate_pos[:, 1])
    xy_dist_total = torch.sqrt(xy_dist_x**2 + xy_dist_y**2)
    z_height = mug_pos[:, 2] - plate_pos[:, 2]
    
    # Check XY overlap
    xy_close = (xy_dist_x < xy_tol) & (xy_dist_y < xy_tol)
    
    # Check Z: mug above plate (within tolerance)
    z_above = z_height > -0.005
    z_close = z_height < z_tol
    
    on_plate = xy_close & z_above & z_close
    
    # Print EVERY 30 steps to see what's happening
    if env.common_step_counter % 30 == 0:
        print(f"\n[DEBUG STEP {env.common_step_counter}]", flush=True)
        print(f"  Mug XYZ: [{mug_pos[0, 0]:.3f}, {mug_pos[0, 1]:.3f}, {mug_pos[0, 2]:.3f}]", flush=True)
        print(f"  Plate XYZ: [{plate_pos[0, 0]:.3f}, {plate_pos[0, 1]:.3f}, {plate_pos[0, 2]:.3f}]", flush=True)
        print(f"  XY dist: {xy_dist_total[0]:.4f}m (need <{xy_tol}m) ✓" if xy_close[0] else f"  XY dist: {xy_dist_total[0]:.4f}m (need <{xy_tol}m) ✗", flush=True)
        print(f"  Z height: {z_height[0]:.4f}m (need 0 to {z_tol}m) ✓" if (z_above[0] and z_close[0]) else f"  Z height: {z_height[0]:.4f}m (need 0 to {z_tol}m) ✗", flush=True)
        print(f"  SUCCESS: {on_plate[0]}", flush=True)
    
    # Also print when actually successful
    if on_plate[0]:
        print(f"[✓ SUCCESS!] XY: {xy_dist_total[0]:.4f}m, Z: {z_height[0]:.4f}m", flush=True)
    
    return on_plate


# def mug_on_plate(
#         env: ManagerBasedRLEnv, 
#         xy_tol: float = 0.05, 
#         z_tol: float = 0.01
#     ) -> torch.Tensor:
#     """
#     Termination: mug is on top of the plate (AABB overlap in XY, mug above plate in Z).
#     Args:
#         env: The environment instance.
#         xy_tol: Tolerance for XY overlap (meters).
#         z_tol: Tolerance for Z height difference (meters).
#     Returns:
#         torch.BoolTensor of shape (num_envs,) indicating termination.
#     """
#     # Get mug and plate root positions (assumes env.scene["mug"] and env.scene["plate"] exist)
#     mug_pos = env.scene["mug"].data.root_pos_w  # (num_envs, 3)
#     plate_pos = env.scene["plate"].data.root_pos_w  # (num_envs, 3)
#     # # DEBUG prints
#     # print(f"[DEBUG] Mug pos: {mug_pos[0]}")
#     # print(f"[DEBUG] Plate pos: {plate_pos[0]}")
#     # Check XY overlap
#     xy_close = (torch.abs(mug_pos[:, 0] - plate_pos[:, 0]) < xy_tol) & (torch.abs(mug_pos[:, 1] - plate_pos[:, 1]) < xy_tol)
#     # Check Z: mug above plate (within tolerance)
#     z_above = (mug_pos[:, 2] - plate_pos[:, 2]) > 0
#     z_close = (mug_pos[:, 2] - plate_pos[:, 2]) < z_tol
#     on_plate = xy_close & z_above & z_close
#     return on_plate
