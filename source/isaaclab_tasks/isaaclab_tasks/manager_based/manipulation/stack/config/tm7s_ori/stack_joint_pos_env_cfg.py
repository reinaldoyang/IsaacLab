# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.tm7s_mdp import tm7s_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.tm7s_original import TM7S_CFG  # isort: skip


@configclass
class EventCfg:
    init_tm7s_arm_pose = EventTerm(
        func=tm7s_stack_events.set_default_tm7s_arm_pose,
        mode="reset",
        params={
            "default_arm_pose": [0.0, 0.0, 1.57, 0.0, 1.57, 0.0],  # Only 6 arm joints
            "arm_joint_names": ["shoulder_1_joint", "shoulder_2_joint", "elbow_joint", 
                                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        },
    )

    randomize_cube_positions = EventTerm(
        func=tm7s_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class TM7SOriginalCubeStackEnvCfg(StackEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events = EventCfg()

        # Set TM7S Original as robot
        self.scene.robot = TM7S_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for TM7S arm (joint-space control)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"], scale=0.5, use_default_offset=True
        )

        # Set actions for Robotiq 2F-85 gripper (comment this part of the code to run test)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "finger_joint",
            ],
            open_command_expr={
                "finger_joint": 0.0,
            },
            close_command_expr={
                "finger_joint": 0.7,
            },
        )
        # utilities for gripper status check
        self.gripper_joint_names = [
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        self.gripper_open_val = 0.0
        self.gripper_close_val = 0.7
        self.gripper_threshold = 0.01

        self.observations.policy.gripper_pos = ObsTerm(
            func=mdp.gripper_drive_joint_pos,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "drive_joint_name": "finger_joint",
            },
        )

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube deterministically
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tm7s_official/body/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tm7s_official/body/flange_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
                # Gripper frame (comment this part of the code to run test env)
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_140_physics_edit/robotiq_base_link",
                    # prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/base_link",
                    name="robotiq_grasp_frame",
                    offset=OffsetCfg(
                        pos=[0.130, 0.0, 0.0],
                        rot=[1.0, 0.0, 0.0, 0.0]
                    ),
                ),
            ],
        )
