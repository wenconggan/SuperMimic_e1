"""Validate robot/motion dimensions and differentiable FK against CPU MuJoCo."""

import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["e1"], default="e1")
parser.add_argument("--motion_dir", default=None)
args = parser.parse_args()
os.environ["SUPERMIMIC_ROBOT"] = args.robot

import mujoco
import numpy as np
import torch

from supermimic.kinematics import RobotKinematics
from supermimic.mdp import MDP, OBS_DIM, STATE_DIM, XI_DIM, flatten_state, unflatten_state
from supermimic.motion import GOAL_FRAME_DIM, MotionLib
from supermimic.robot import JOINT_NAMES, MJCF_PATH, NUM_J, TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import mat_to_quat


def main():
    torch.manual_seed(0)
    kin = RobotKinematics("cpu")
    motion_dir = args.motion_dir or str(REPO / "datasets" / "e1_lafan")
    motion = MotionLib.from_dir(motion_dir, kin, "cpu")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    assert JOINT_NAMES == [model.joint(i).name for i in range(1, model.njnt)]
    frames = torch.linspace(0, motion.num_frames - 1, 16).long()
    S = list(motion.ref_state(frames))
    # Nonzero independent velocities catch frame/convention and fixed-link bugs.
    S[2], S[3], S[5] = torch.randn(16, 3), torch.randn(16, 3), torch.randn(16, NUM_J)
    body_p, body_R, body_v, body_w = kin.forward(*S)
    ids = [model.body(name).id for name in TRACKED_BODIES]
    errors = np.zeros(4)
    for i in range(len(frames)):
        data = mujoco.MjData(model)
        data.qpos[:3] = S[0][i].numpy()
        data.qpos[3:7] = mat_to_quat(S[1][i]).numpy()
        data.qpos[7:] = S[4][i].numpy()
        data.qvel[:3] = S[2][i].numpy()
        data.qvel[3:6] = (S[1][i].T @ S[3][i]).numpy()
        data.qvel[6:] = S[5][i].numpy()
        mujoco.mj_forward(model, data)
        vel = np.zeros((len(ids), 6))
        for k, body_id in enumerate(ids):
            # XBODY refers to the link origin; BODY refers to its inertial frame.
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, vel[k], 0)
        pairs = ((body_p[i], data.xpos[ids]), (body_R[i], data.xmat[ids].reshape(-1, 3, 3)),
                 (body_v[i], vel[:, 3:]), (body_w[i], vel[:, :3]))
        errors = np.maximum(errors, [np.abs(a.numpy() - b).max() for a, b in pairs])
    assert errors.max() < 1e-5, f"FK mismatch: {errors}"
    flat = flatten_state(S)
    assert flat.shape == (16, STATE_DIM)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(S, unflatten_state(flat)))
    obs = mdp.obs(S, torch.zeros(16, NUM_J), torch.zeros(16, XI_DIM), frames)
    assert obs.shape == (16, OBS_DIM)
    assert motion.goal(frames, 10).shape == (16, 10 * GOAL_FRAME_DIM)
    assert (motion.bin_start < motion.clip_end[motion.bin_clip]).all(), "terminal-only reset bin"
    test_S = tuple(x.clone().requires_grad_() for x in motion.ref_state(frames))
    reward = mdp.reward(test_S, frames, torch.zeros(16, NUM_J), torch.zeros(16, NUM_J))
    reward.sum().backward()
    assert torch.isfinite(reward).all() and all(torch.isfinite(x.grad).all() for x in test_S)
    print(f"PASS robot={args.robot}: {len(motion.names)} clips, {motion.num_frames} frames, {motion.num_frames * .02 / 3600:.2f} h")
    print(f"joints={NUM_J}, state={STATE_DIM}, observation={OBS_DIM}, context={XI_DIM}, goal={10 * GOAL_FRAME_DIM}")
    print(f"FK max errors (position, rotation, linear/angular velocity): {errors.tolist()}")
    print(f"motion joint-limit violations: {((motion.q < kin.q_lo) | (motion.q > kin.q_hi)).float().mean().item():.3%}")


if __name__ == "__main__":
    main()
