from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_assets.robots.tm7s_original import TM7S_CFG
from . import stack_joint_pos_env_cfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.tm7s_mdp import tm7s_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.tm7s_mdp.fixed_pose_ik_action import FixedPoseIKActionCfg


@configclass
class TM7SOriginalObservationsCfg:
    """Observation specifications for TM7S Original IK-Rel policy (matching robomimic training config)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - must match bc_rnn_low_dim.json modalities."""

        # Only include the 4 observations defined in bc_rnn_low_dim.json:
        # "low_dim": ["eef_pos", "eef_quat", "gripper_pos", "object"]
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(
            func=mdp.tm7s_gripper_pos,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "drive_joint_name": "finger_joint",
                "open_q": 0.0,
                "close_q": 0.5,
            },
        )
        object = ObsTerm(func=mdp.object_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
                "ee_target_idx": 1,          # robotiq_grasp_frame
                "diff_threshold": 0.08,
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
                "ee_target_idx": 1,          # robotiq_grasp_frame
                "diff_threshold": 0.08,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TM7SOriginalCubeStackEnvCfg(stack_joint_pos_env_cfg.TM7SOriginalCubeStackEnvCfg):
    # Override observations to match robomimic training config
    observations: TM7SOriginalObservationsCfg = TM7SOriginalObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        
        #Set TM7S as robot
        self.scene.robot = TM7S_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Add event to initialize IK target to current EE pose at reset
        # This prevents drift when starting teleop with no key presses
        # self.events.init_ik_target = EventTerm(
        #     func=tm7s_stack_events.init_ik_target_to_current_ee_pose,
        #     mode="reset",
        # )
        
        # USE FIXED POSE IK FOR TESTING - ignores input actions, always targets reset pose
        # Comment this out and uncomment the regular DifferentialInverseKinematicsActionCfg below for normal operation
        # self.actions.arm_action = FixedPoseIKActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         "shoulder_1_joint",
        #         "shoulder_2_joint",
        #         "elbow_joint",
        #         "wrist_1_joint",
        #         "wrist_2_joint",
        #         "wrist_3_joint",
        #     ],
        #     body_name="flange_link",
        #     controller=DifferentialIKControllerCfg(
        #         command_type="pose", use_relative_mode=True, ik_method="dls",
        #     ),
        #     scale=0.5,
        #     body_offset=FixedPoseIKActionCfg.OffsetCfg(
        #         pos=[0.0, 0.0, 0.0],
        #     ),
        # )
        
        # NORMAL IK ACTION - uncomment for regular teleop operation
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_1_joint",
                "shoulder_2_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="flange_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0],
            ),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )