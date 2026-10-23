"""Native MuJoCo conventions that must agree with the trained E1 policy."""

import os
from pathlib import Path
import sys

os.environ["SUPERMIMIC_ROBOT"] = "e1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
import pytest
import torch

from supermimic.robot import JOINT_NAMES, MJCF_PATH, default_joint_pos_list, drive_parameters
from supermimic.rotations import so3_exp
from tools.sim2sim import NativeRobot, build_model


@pytest.fixture
def robot():
    # Reversing actuator creation order detects a reliance on positional indices.
    model = build_model(MJCF_PATH, list(reversed(JOINT_NAMES)), drive_parameters, 4)
    return NativeRobot(model, JOINT_NAMES, drive_parameters)


def state():
    return (torch.tensor([[0.4, -0.3, 1.25]]),
            so3_exp(torch.tensor([[0.3, -0.2, 0.4]])),
            torch.tensor([[0.4, 0.2, -0.3]]), torch.tensor([[-0.7, 0.2, 0.6]]),
            torch.tensor([default_joint_pos_list()]), torch.linspace(-0.3, 0.3, len(JOINT_NAMES))[None])


def test_root_link_state_roundtrip_with_rotation_and_motion(robot):
    expected = state()
    robot.reset(expected)
    for got, want in zip(robot.state(), expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=2e-6)


def test_named_actuators_action_scale_and_torque_limits(robot):
    robot.reset(state())
    action = np.linspace(-1000, 1000, len(JOINT_NAMES))
    default = np.array(default_joint_pos_list())
    robot.step(action, default, 1)
    for i, name in enumerate(JOINT_NAMES):
        kp, kd, limit, _, _ = drive_parameters(name)
        act = robot.model.actuator(f"policy_{name}").id
        assert robot.data.ctrl[act] == pytest.approx(default[i] + 0.25 * limit / kp * action[i])
        assert abs(robot.data.actuator_force[act]) <= limit + 1e-8
    assert np.isfinite(robot.data.qpos).all()


def test_ground_contacts_remain_with_self_collisions_disabled(robot):
    S = list(state())
    S[0] = torch.tensor([[0.0, 0.0, 0.2]])
    S[1] = torch.eye(3)[None]
    robot.reset(S)
    assert robot.data.ncon > 0
    for contact in robot.data.contact:
        bodies = robot.model.geom_bodyid[[contact.geom1, contact.geom2]]
        assert 0 in bodies, f"Unexpected robot self contact between bodies {bodies}"
