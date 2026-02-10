# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Simple arm-only test environment for TM7S Original (no gripper).

This is a minimal test environment to validate robot stability under joint-position control.
Run with zero actions to verify the robot stays still at its home pose.
"""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.tm7s_original import TM7S_CFG
from isaaclab_tasks.manager_based.manipulation.stack.tm7s_mdp import tm7s_stack_events


# Define the home pose for TM7S arm (6 joints)
TM7S_HOME_POSE = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
TM7S_ARM_JOINT_NAMES = [
    "shoulder_1_joint",
    "shoulder_2_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


@configclass
class ArmTestSceneCfg(InteractiveSceneCfg):
    """Scene configuration for TM7S Original arm test."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)),
    )

    # robot
    robot: ArticulationCfg = TM7S_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # end-effector frame
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tm7s_official/body/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/tm7s_official/body/flange_link",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the arm test environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for arm test (arm only, no gripper)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,
    )

    # Robotiq 85 gripper action
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.8},
    )


@configclass
class EventsCfg:
    """Events for the arm test environment - reset to exact home pose (no randomization)."""

    reset_robot = EventTerm(
        func=tm7s_stack_events.set_default_tm7s_arm_pose,
        mode="reset",
        params={
            "default_arm_pose": TM7S_HOME_POSE,
            "arm_joint_names": TM7S_ARM_JOINT_NAMES,
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions for the arm test environment."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """Reward terms for the arm test environment (dummy, just for compatibility)."""

    # Small penalty for action magnitude to encourage smooth movements
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TM7SOriginalArmTestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the TM7S Original arm test environment (no gripper)."""

    # Scene settings
    scene: ArmTestSceneCfg = ArmTestSceneCfg(num_envs=1, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
