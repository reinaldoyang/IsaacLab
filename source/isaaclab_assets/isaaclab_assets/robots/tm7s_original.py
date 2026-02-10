"""TM7S robot articulation configuration.

This mirrors the style used by `franka.py` so the configuration is
consistent with other robot configs in the repo.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


TM7S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/reinaldoyang/Documents/USD_files/tm7s_official_robotiq140.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity = 5.0, #when two objects penetrate each other, this limits the velocity they will be separated at
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count = 8, solver_velocity_iteration_count = 0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos = {
            "shoulder_1_joint": 0.0,
            "shoulder_2_joint": 0.0,
            "elbow_joint": 1.57,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 1.57,
            "wrist_3_joint": 0.0,

            # Robotiq gripper joints
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            # "left_inner_finger_knuckle_joint": 0.0,
            # "right_inner_finger_knuckle_joint": 0.0,
            #robotiq140
            "left_inner_finger_pad_joint": 0.0,
            "right_inner_finger_pad_joint": 0.0,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_.*", "wrist_.*"],
            effort_limit_sim=30000,
            velocity_limit_sim=158,
            stiffness=1000000.0,
            damping=10000.0,
        ),
        # Gripper actuators
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=11.25,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_joint",
                "right_inner_finger_joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.2,
            damping=0.01,
            friction=0.0,
            armature=0.0,
        ),
        # Outer knuckle joints as passive (no PD)
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_outer_knuckle_joint",
                # "left_inner_finger_knuckle_joint",
                # "right_inner_finger_knuckle_joint",
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.01,
            friction=0.0,
            armature=0.0,
        ),
        # # Gripper actuators
        # "gripper": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "finger_joint",
        #     ],
        #     velocity_limit_sim=0.5,
        #     effort_limit_sim=1000.0,
        #     stiffness=50.0,
        #     damping=1.0,
        # ),
        # "gripper_finger": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "left_inner_finger_joint",
        #         "right_inner_finger_joint",
        #     ],
        #     effort_limit_sim=50.0,
        #     velocity_limit_sim=100.0,
        #     stiffness=0.2,
        #     damping=0.001,
        # ),
        # # Outer knuckle joints as passive (no PD)
        # "gripper_passive": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "right_outer_knuckle_joint",
        #         # "left_inner_finger_knuckle_joint",
        #         # "right_inner_finger_knuckle_joint",
        #         "left_inner_finger_pad_joint",
        #         "right_inner_finger_pad_joint",
        #     ],
        #     effort_limit_sim=1.0,
        #     velocity_limit_sim=100.0,
        #     stiffness=0.0,
        #     damping=0.0,
        # ),
    },
    soft_joint_pos_limit_factor=1.0,
)


