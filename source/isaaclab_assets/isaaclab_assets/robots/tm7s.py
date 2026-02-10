"""TM7S robot articulation configuration.

This mirrors the style used by `franka.py` so the configuration is
consistent with other robot configs in the repo.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


TM7S_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/reinaldoyang/Documents/USD_files/tm7s_robotiq85_forIsaacLab_flatten_20260204.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity = 5.0, #when two objects penetrate each other, this limits the velocity they will be separated at
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count = 64, solver_velocity_iteration_count = 4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Replace/add your robot joint names here to set sensible defaults
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 1.57,
            "joint_4": 0.0,
            "joint_5": 1.57,
            "joint_6": 0.0,  # Gripper faces down
            # Robotiq 2F-85 gripper joints (explicit to avoid regex overlap)
            "robotiq_85_left_inner_knuckle_joint": 0.0,
            "robotiq_85_right_inner_knuckle_joint": 0.0,
            "robotiq_85_left_finger_tip_joint": 0.0,
            "robotiq_85_right_finger_tip_joint": 0.0,
            "robotiq_85_left_knuckle_joint": 0.0,
            "robotiq_85_right_knuckle_joint": 0.0,
        }
    ),
    actuators={
        "tm7s_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-3]"],
            effort_limit_sim=30000,
            velocity_limit_sim=1518,
            stiffness=1000000.0,
            damping=100000.0,
        ),
        "tm7s_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_[4-6]"],
            effort_limit_sim=30000,
            velocity_limit_sim=1518,
            stiffness=1000000.0,
            damping=100000.0,
        ),
        # Main gripper drive joints (inner knuckles)
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_85_left_knuckle_joint",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=0.5,
            stiffness=50.0,
            damping=1.0,
        ),
        # Finger tips that follow the drive
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_85_left_inner_knuckle_joint",
                "robotiq_85_right_inner_knuckle_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=100.0,
            stiffness=20,
            damping=0.001,
        ),
        # Outer knuckle joints as passive (no PD)
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_85_left_finger_tip_joint",
                "robotiq_85_right_finger_tip_joint",
                "robotiq_85_right_knuckle_joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=100.0,
            stiffness=30.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


