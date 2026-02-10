# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
IK Frame Alignment Test for TM7S robot.

This script verifies that:
1. The IK body_name matches an actual body in the robot
2. The ee_frame sensor is properly configured
3. The IK controller maintains pose when given zero commands

Usage:
    ./isaaclab.sh -p scripts/environments/test_tm7s_ik_alignment.py --task TM7S-Original-Cube-Stack --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="TM7S IK Frame Alignment Test")
parser.add_argument("--task", type=str, default="TM7S-Original-Cube-Stack", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps to run.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    # Get references
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    action_manager = env.action_manager
    
    print("=" * 70)
    print("TM7S IK FRAME ALIGNMENT TEST")
    print("=" * 70)
    
    # 1. Print robot body names to verify IK body exists
    print("\n1. ROBOT BODY NAMES:")
    print(f"   Total bodies: {robot.num_bodies}")
    for i, name in enumerate(robot.body_names):
        marker = " <-- IK target" if "flange" in name.lower() else ""
        print(f"   [{i}] {name}{marker}")
    
    # 2. Check ee_frame configuration
    print("\n2. EE_FRAME SENSOR:")
    print(f"   Source frame: {ee_frame.cfg.prim_path}")
    print(f"   Target frames:")
    for tf in ee_frame.cfg.target_frames:
        print(f"     - {tf.name}: {tf.prim_path}")
        print(f"       offset: pos={tf.offset.pos}, rot={tf.offset.rot if hasattr(tf.offset, 'rot') else 'default'}")
    
    # 3. Check IK action configuration
    print("\n3. ACTION MANAGER:")
    print(f"   Total action dim: {action_manager.total_action_dim}")
    ik_term = None
    for term_name, term in action_manager._terms.items():
        print(f"   '{term_name}':")
        print(f"     action_dim: {term.action_dim}")
        if hasattr(term, '_joint_ids'):
            print(f"     joint_ids: {term._joint_ids}")
        if hasattr(term, 'cfg'):
            cfg = term.cfg
            if hasattr(cfg, 'body_name'):
                print(f"     body_name: {cfg.body_name}")
                ik_term = term  # Save IK term for later inspection
            if hasattr(cfg, 'controller'):
                ctrl = cfg.controller
                print(f"     controller: command_type={ctrl.command_type}, relative_mode={ctrl.use_relative_mode}")
    
    # 3b. Check IK controller's internal state BEFORE reset
    print("\n3b. IK CONTROLLER INITIAL STATE (before reset):")
    if ik_term is not None and hasattr(ik_term, '_ik_controller'):
        ik_ctrl = ik_term._ik_controller
        print(f"   ee_pos_des (initial): {ik_ctrl.ee_pos_des[0].tolist()}")
        print(f"   ee_quat_des (initial): {ik_ctrl.ee_quat_des[0].tolist()}")
        if ik_ctrl.ee_quat_des[0].norm().item() < 0.001:
            print("   ⚠️  WARNING: IK target quaternion is zero! This will cause issues!")
    else:
        print("   Could not access IK controller")
    
    # Reset environment
    obs, _ = env.reset()
    
    # 3c. Check IK controller's internal state AFTER reset but BEFORE any action
    print("\n3c. IK CONTROLLER STATE (after reset, before any action):")
    if ik_term is not None and hasattr(ik_term, '_ik_controller'):
        ik_ctrl = ik_term._ik_controller
        print(f"   ee_pos_des: {ik_ctrl.ee_pos_des[0].tolist()}")
        print(f"   ee_quat_des: {ik_ctrl.ee_quat_des[0].tolist()}")
        quat_norm = ik_ctrl.ee_quat_des[0].norm().item()
        print(f"   ee_quat_des norm: {quat_norm:.6f}")
        if quat_norm < 0.001:
            print("   ❌ CRITICAL: IK target quaternion is still zero after reset!")
            print("      This means the first action will compute error from origin!")
    
    # 3d. FIX: Compute current EE pose and set IK target to it
    print("\n3d. INITIALIZING IK TARGET TO CURRENT EE POSE:")
    if ik_term is not None and hasattr(ik_term, '_ik_controller'):
        # Use the IK action term's own method to compute frame pose (in root frame)
        ee_pos_curr, ee_quat_curr = ik_term._compute_frame_pose()
        print(f"   Current EE pose (in root frame):")
        print(f"     pos: {ee_pos_curr[0].tolist()}")
        print(f"     quat: {ee_quat_curr[0].tolist()}")
        
        # Set the IK controller's target to current pose
        ik_ctrl = ik_term._ik_controller
        ik_ctrl.ee_pos_des[:] = ee_pos_curr
        ik_ctrl.ee_quat_des[:] = ee_quat_curr
        
        print(f"   ✅ Set IK target to current EE pose!")
        print(f"   ee_pos_des (after fix): {ik_ctrl.ee_pos_des[0].tolist()}")
        print(f"   ee_quat_des (after fix): {ik_ctrl.ee_quat_des[0].tolist()}")
    
    # 4. Get initial EE pose from both robot body state and ee_frame sensor
    print("\n4. INITIAL EE POSE COMPARISON:")
    
    # From ee_frame sensor
    ee_pos_sensor = ee_frame.data.target_pos_w[:, 0, :]  # First target frame (end_effector)
    ee_quat_sensor = ee_frame.data.target_quat_w[:, 0, :]
    print(f"   From ee_frame sensor:")
    print(f"     pos: {ee_pos_sensor[0].tolist()}")
    print(f"     quat: {ee_quat_sensor[0].tolist()}")
    
    # Try to find flange_link in body_names
    try:
        flange_idx = robot.body_names.index("flange_link")
        ee_pos_body = robot.data.body_pos_w[:, flange_idx, :]
        ee_quat_body = robot.data.body_quat_w[:, flange_idx, :]
        print(f"   From robot body state (flange_link):")
        print(f"     pos: {ee_pos_body[0].tolist()}")
        print(f"     quat: {ee_quat_body[0].tolist()}")
        
        # Check alignment
        pos_diff = (ee_pos_sensor - ee_pos_body).norm().item()
        print(f"   Position difference: {pos_diff:.6f} m")
        if pos_diff < 0.001:
            print("   ✅ EE frame and body state are aligned")
        else:
            print("   ⚠️  EE frame and body state have some offset")
    except ValueError:
        print("   ⚠️  'flange_link' not found in body names!")
        print(f"   Available bodies: {robot.body_names}")
    
    # 5. Test IK stability with zero commands
    print("\n5. IK STABILITY TEST (zero commands):")
    
    # Record initial joint positions
    initial_joint_pos = robot.data.joint_pos.clone()
    initial_ee_pos = ee_pos_sensor.clone()
    
    # Create zero action
    zero_action = torch.zeros(env.num_envs, action_manager.total_action_dim, device=env.device)
    
    # First step - check what happens
    print("\n   FIRST STEP ANALYSIS:")
    obs, _, _, _, _ = env.step(zero_action)
    
    # Check IK controller state after first step
    if ik_term is not None and hasattr(ik_term, '_ik_controller'):
        ik_ctrl = ik_term._ik_controller
        print(f"   IK ee_pos_des after step 1: {ik_ctrl.ee_pos_des[0].tolist()}")
        print(f"   IK ee_quat_des after step 1: {ik_ctrl.ee_quat_des[0].tolist()}")
        
        # Compare with actual EE pose
        current_ee_pos = ee_frame.data.target_pos_w[:, 0, :]
        pos_error = (ik_ctrl.ee_pos_des[0] - current_ee_pos[0]).norm().item()
        print(f"   Position error (IK target vs actual EE): {pos_error:.6f} m")
        if pos_error > 0.1:
            print("   ❌ CRITICAL: Large position error - IK target is far from actual EE!")
    
    print(f"\n   Running {args_cli.num_steps - 1} more steps with ZERO actions...")
    
    max_ee_drift = 0.0
    max_joint_drift = torch.zeros(robot.num_joints, device=env.device)
    
    for step in range(1, args_cli.num_steps):  # Start from 1 since we already did step 0
        obs, _, _, _, _ = env.step(zero_action)
        
        # Track EE drift
        current_ee_pos = ee_frame.data.target_pos_w[:, 0, :]
        ee_drift = (current_ee_pos - initial_ee_pos).norm().item()
        max_ee_drift = max(max_ee_drift, ee_drift)
        
        # Track joint drift
        current_joint_pos = robot.data.joint_pos
        joint_drift = (current_joint_pos - initial_joint_pos).abs()
        max_joint_drift = torch.max(max_joint_drift, joint_drift[0])
        
        if (step + 1) % 100 == 0:
            print(f"   Step {step + 1}: EE drift = {ee_drift:.6f} m, Joint drift sum = {joint_drift[0].sum().item():.6f} rad")
    
    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    
    final_ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    final_joint_pos = robot.data.joint_pos
    
    print(f"\nEE Position drift: {(final_ee_pos - initial_ee_pos).norm().item():.6f} m (max: {max_ee_drift:.6f} m)")
    print(f"Total joint drift: {(final_joint_pos - initial_joint_pos).abs().sum().item():.6f} rad")
    
    print("\nPer-joint drift:")
    arm_joints = ["shoulder_1_joint", "shoulder_2_joint", "elbow_joint", 
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    for i, name in enumerate(robot.joint_names):
        drift = max_joint_drift[i].item()
        is_arm = any(aj in name for aj in arm_joints)
        marker = " (arm)" if is_arm else " (gripper)"
        if drift > 0.001:
            print(f"  ⚠️  {name:40s}: max_drift = {drift:.6f} rad{marker}")
        else:
            print(f"  ✅ {name:40s}: max_drift = {drift:.6f} rad{marker}")
    
    if max_ee_drift < 0.01:
        print("\n✅ PASS: IK controller is stable (EE drift < 0.01 m)")
    elif max_ee_drift < 0.05:
        print("\n⚠️  WARNING: Some EE drift detected (0.01 < drift < 0.05 m)")
    else:
        print("\n❌ FAIL: Significant EE drift (> 0.05 m)")
        print("   Check IK body_name, ee_frame config, and controller settings")
    
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
