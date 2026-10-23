"""E1 21-DoF articulation, joint layout, and trained position-servo parameters.

The MJCF and CSV files share the same joint order. Isaac Lab and native MuJoCo
use the home pose, PD gains, effort limits, armatures, and action scales here.
"""

from __future__ import annotations

import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_NAME = os.environ.get("SUPERMIMIC_ROBOT", "e1")
if ROBOT_NAME != "e1":
    raise ValueError(f"supermimic_e1 supports only E1, got {ROBOT_NAME!r}")
DATA_DIR = os.path.join(REPO_DIR, "data")
MJCF_PATH = os.path.join(REPO_DIR, "robots", "e1_21dof", "mjcf", "E1_21dof.xml")
URDF_PATH = os.path.join(REPO_DIR, "robots", "e1_21dof", "urdf", "E1_21dof.urdf")
USD_DIR = os.path.join(DATA_DIR, "e1_usd")
USD_PATH = os.path.join(USD_DIR, "e1_21dof", "e1_21dof.usda")
usd_manifest = os.path.join(USD_DIR, "asset_path.txt")
if os.path.isfile(usd_manifest):
    with open(usd_manifest) as manifest:
        USD_PATH = os.path.join(USD_DIR, manifest.read().strip())


# Joint order of the supplied E1 LAFAN1 retarget CSVs.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_pitch_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_pitch_link",
]  # fmt: skip

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.10, ".*_knee_joint": 0.23, ".*_ankle_pitch_joint": -0.13,
    "left_shoulder_roll_joint": 0.25, "right_shoulder_roll_joint": -0.25, ".*_elbow_joint": 1.0,
}


def default_joint_pos_list() -> list[float]:
    import re

    out = []
    for name in JOINT_NAMES:
        val = 0.0
        for pat, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pat, name):
                val = v
        out.append(val)
    return out


# (Kp, Kd, torque limit, velocity rating), from the supplied E1 asset configuration.
E1_DRIVES = {
    "hip_pitch": (200., 5., 120., 12.04), "hip_roll": (200., 5., 120., 12.04),
    "hip_yaw": (80., 3., 36., 13.61), "knee": (200., 5., 120., 12.04),
    "ankle_pitch": (80., 3., 36., 13.61), "ankle_roll": (60., 2., 30., 15.71),
    "waist_yaw": (150., 4., 60., 13.09), "shoulder_pitch": (30., 2., 60., 13.09),
    "shoulder_roll": (30., 2., 36., 13.61), "shoulder_yaw": (30., 2., 15., 15.71),
    "elbow": (30., 2., 60., 13.09),
}

E1_JOINT_NAMES = JOINT_NAMES
G1_JOINT_NAMES = JOINT_NAMES  # Compatibility with upstream imports.
NUM_J = len(JOINT_NAMES)
NUM_TRACKED = len(TRACKED_BODIES)


def drive_parameters(joint_name: str) -> tuple[float, float, float, float, float]:
    """Kp, Kd, effort limit, velocity rating, and armature for a named joint."""
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    kp, kd, effort, velocity = E1_DRIVES[key]
    return kp, kd, effort, velocity, 0.01


def action_scale(joint_name: str) -> float:
    kp, _, effort, _, _ = drive_parameters(joint_name)
    return 0.25 * effort / kp


def _actuators():
    from isaaclab.actuators import ImplicitActuatorCfg
    acts = {}
    for name in JOINT_NAMES:
        kp, kd, effort, vel, arm = drive_parameters(name)
        acts[name.replace("_joint", "")] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=effort,
            velocity_limit_sim=vel,
            stiffness=kp,
            damping=kd,
            armature=arm,
        )
    return acts


def _isaaclab_cfg():
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg

    if not os.path.isfile(USD_PATH):
        raise FileNotFoundError(
            f"E1 USD asset not found: {USD_PATH}. Run python tools/mjcf_to_usd.py "
            "before training; see install.md. Native MuJoCo sim2sim uses the MJCF directly."
        )
    return ArticulationCfg(
        # Import offline: schema plugins cannot register in a running Isaac Lab process.
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            variants={"Physics": "mujoco"},
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),
            joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=_actuators(),
    )


def __getattr__(name):
    # Keep the original Isaac Lab entry points available without importing Isaac Lab
    # when loading motions or networks for offline checks.
    if name in ("E1_CFG", "G1_CFG"):
        return _isaaclab_cfg()
    raise AttributeError(name)
