# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.scene import InteractiveSceneCfg
import math


from isaaclab_tasks.manager_based.manipulation.pick_place_arm import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place_arm.mdp import pnp_events
from isaaclab_tasks.manager_based.manipulation.pick_place_arm.pnp_plate_env_cfg import PnpEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
@configclass
class FrankaPlateSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with TM7S robot and table."""

    # Robot - TM7S with Robotiq 2F-85 gripper
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    # Mug object (will be set in __post_init__)
    mug: RigidObjectCfg = MISSING

    # End-effector frame sensor (will be set in __post_init__)
    ee_frame: FrameTransformerCfg = MISSING


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=pnp_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=pnp_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_mug= EventTerm(
        func=pnp_events.randomize_object_pose,  # or import from stack if you want
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("mug")],
            "pose_range": {
                "x": (0.6, 0.7),  # adjust as needed
                "y": (-0.2, 0.2),
                "z": (0.05, 0.05),  # keep on table
                "yaw": (math.pi/2, math.pi/2),
            },
            "min_separation": 1,  # adjust as needed
        },
    )

    randomize_plate = EventTerm(
        func=pnp_events.randomize_object_pose,  # or import from stack if you want
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("plate")],
            "pose_range": {
                "x": (0.4, 0.5),  # adjust as needed
                "y": (-0.2, 0.2),
                "z": (0.05, 0.05),  # keep on table
                "roll": (math.pi/2, math.pi/2),
            },
            "min_separation": 1,  # adjust as needed
        },
    )


@configclass
class FrankaPlateEnvCfg(PnpEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Plate object on the table (copied from tm7s_base_env_cfg)
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
        plate_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        mug_properties = RigidBodyPropertiesCfg(
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
                rot=[0.7071, 0.0, 0.7071, 0.0],  # +90° around X axis
            ),
            spawn=UsdFileCfg(
                usd_path="/home/reinaldoyang/IsaacLab/plate.usd",
                scale=(0.01, 0.01, 0.01),
                rigid_props=plate_properties,
                semantic_tags=[("class", "plate")],
            ),
        )
        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Mug",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.8, 0.0, 0.05],  # place it slightly above the table
                rot=[0.707, 0.0, 0.0, 0.7071],  # rotate if needed
            ),
            spawn=UsdFileCfg(
                usd_path="/home/reinaldoyang/IsaacLab/mug.usd",  # adjust path to your mug asset
                scale=(0.01, 0.01, 0.01),  # scale appropriately
                rigid_props=mug_properties,
                semantic_tags=[("class", "mug")],
            ),
        )
        

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )