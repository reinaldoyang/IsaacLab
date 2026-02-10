# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple stability test for TM7S robot.

This script tests whether the robot stays still when given zero actions.
If the robot drifts, collapses, or rotates, there's likely an issue with:
- Drive gains (stiffness/damping)
- USD joint constraints
- Initial state configuration

Usage:
    ./isaaclab.sh -p scripts/environments/test_tm7s_stability.py --task TM7S-Original-Arm-Test --num_envs 1
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="TM7S Stability Test - Zero Action Test")
parser.add_argument("--task", type=str, default="TM7S-Original-Arm-Test", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps to run.")

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
    
    # Get robot reference
    robot = env.scene["robot"]
    
    # Print robot metadata
    print("=" * 70)
    print("TM7S STABILITY TEST - ROBOT METADATA")
    print("=" * 70)
    print(f"Number of joints: {robot.num_joints}")
    print(f"Joint names: {robot.joint_names}")
    print(f"Number of bodies: {robot.num_bodies}")
    print(f"Body names: {robot.body_names}")
    print()
    
    # Print actuator info
    print("ACTUATORS:")
    for name, actuator in robot.actuators.items():
        print(f"  '{name}': joints={actuator.joint_names}, num_joints={actuator.num_joints}")
    print()
    
    # Print action manager info
    print("ACTION MANAGER:")
    action_manager = env.action_manager
    print(f"  Total action dim: {action_manager.total_action_dim}")
    for term_name, term in action_manager._terms.items():
        joint_names = getattr(term, '_joint_names', getattr(term, 'joint_names', 'N/A'))
        print(f"  '{term_name}': action_dim={term.action_dim}, joint_names={joint_names}")
    print("=" * 70)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Record initial joint positions
    initial_joint_pos = robot.data.joint_pos.clone()
    print("\nINITIAL JOINT POSITIONS:")
    for i, name in enumerate(robot.joint_names):
        print(f"  {name}: {initial_joint_pos[0, i].item():.6f}")
    print()
    
    # Create zero action (robot should stay still)
    zero_action = torch.zeros(env.num_envs, action_manager.total_action_dim, device=env.device)
    
    print(f"\nRunning {args_cli.num_steps} steps with ZERO actions...")
    print("Expected: Robot should stay essentially still.\n")
    
    # Run simulation
    max_drift = torch.zeros(robot.num_joints, device=env.device)
    
    for step in range(args_cli.num_steps):
        # Step with zero action
        obs, _, _, _, _ = env.step(zero_action)
        
        # Compute drift from initial position
        current_pos = robot.data.joint_pos
        drift = (current_pos - initial_joint_pos).abs()
        max_drift = torch.max(max_drift, drift[0])
        
        # Print every 100 steps
        if (step + 1) % 100 == 0:
            total_drift = drift[0].sum().item()
            print(f"Step {step + 1}: Total drift = {total_drift:.6f} rad")
    
    # Final report
    print("\n" + "=" * 70)
    print("FINAL DRIFT REPORT (after {} steps)".format(args_cli.num_steps))
    print("=" * 70)
    
    final_joint_pos = robot.data.joint_pos.clone()
    
    print("\nPer-joint drift (initial -> final):")
    for i, name in enumerate(robot.joint_names):
        init_val = initial_joint_pos[0, i].item()
        final_val = final_joint_pos[0, i].item()
        drift_val = max_drift[i].item()
        print(f"  {name:40s}: {init_val:+.6f} -> {final_val:+.6f}  (max drift: {drift_val:.6f})")
    
    total_max_drift = max_drift.sum().item()
    print(f"\nTotal max drift: {total_max_drift:.6f} rad")
    
    if total_max_drift < 0.01:
        print("\n✅ PASS: Robot is stable (drift < 0.01 rad)")
    elif total_max_drift < 0.1:
        print("\n⚠️  WARNING: Robot has some drift (0.01 < drift < 0.1 rad)")
        print("   Consider increasing stiffness or checking USD constraints.")
    else:
        print("\n❌ FAIL: Robot is unstable (drift > 0.1 rad)")
        print("   Check drive gains, USD constraints, and initial state.")
    
    print("=" * 70)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
