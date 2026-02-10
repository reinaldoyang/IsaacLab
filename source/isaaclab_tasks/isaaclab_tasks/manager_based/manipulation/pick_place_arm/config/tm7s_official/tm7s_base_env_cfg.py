# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TM7S Base Environment Configuration - Robot and Table only."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.tm7s import TM7S_CFG  # isort: skip


##
# Scene definition
##
@configclass
class TM7STableSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with TM7S robot and table."""

    # Robot - TM7S with Robotiq 2F-85 gripper
    robot: ArticulationCfg = TM7S_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Plate object (will be set in __post_init__)
    plate: RigidObjectCfg = MISSING

    # End-effector frame sensor (will be set in __post_init__)
    ee_frame: FrameTransformerCfg = MISSING


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: JointPositionActionCfg = MISSING
    gripper_action: BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        gripper_pos = ObsTerm(
            func=stack_mdp.gripper_drive_joint_pos,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "drive_joint_name": "robotiq_85_left_knuckle_joint",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Event specifications for the MDP."""

    init_robot_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 1.57, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    randomize_robot_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##
@configclass
class TM7SBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the TM7S base environment (robot + table only)."""

    # Scene settings
    scene: TM7STableSceneCfg = TM7STableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # XR settings for teleoperation
    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 5
        self.episode_length_s = 30.0

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Add semantic tags
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for TM7S arm (joint-space control)
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # Set actions for Robotiq 2F-85 gripper
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "robotiq_85_left_knuckle_joint",
                "robotiq_85_right_knuckle_joint",
            ],
            open_command_expr={
                "robotiq_85_left_knuckle_joint": 0.0,
                "robotiq_85_right_knuckle_joint": 0.0,
            },
            close_command_expr={
                "robotiq_85_left_knuckle_joint": 0.7,
                "robotiq_85_right_knuckle_joint": 0.7,
            },
        )

        # Gripper status check utilities
        self.gripper_joint_names = [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
        ]
        self.gripper_open_val = 0.0
        self.gripper_close_val = 0.4
        self.gripper_threshold = 0.01

        # Plate object on the table
        plate_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        self.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, 0.0],  # Same height as robot base (z=0)
                rot=[0.707, 0.707, 0, 0],  # +90° around X axis
            ),
            spawn=UsdFileCfg(
                usd_path="/home/reinaldoyang/IsaacLab/plate.usd",
                scale=(0.01, 0.01, 0.01),
                rigid_props=plate_properties,
                semantic_tags=[("class", "plate")],
            ),
        )

        # End-effector frame transformer (same as stack config)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/link_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tm7s_robotiq/robotiq_85_base_link",
                    name="robotiq_grasp_frame",
                    offset=OffsetCfg(
                        pos=[0.130, 0.0, 0.0],
                        rot=[1.0, 0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
