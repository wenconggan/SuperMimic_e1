"""Run an E1 SuperMimic checkpoint in standalone CPU MuJoCo, with viewer or MP4.

The policy drives the simulated joints; reference motion is used only for its
goal, initial state, metrics, and the optional blue reference skeleton.
No Isaac Lab, Newton, or world model is needed by this executable.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from collections import deque
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parent.parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path,
                   help="select checkpoints only from this run; default: search logs/e1_paper and models")
    p.add_argument("--checkpoint", "--ckpt", type=Path,
                   help="explicit checkpoint; default: highest iteration among the searched directories")
    p.add_argument("--motion_dir", type=Path, default=REPO / "datasets" / "e1_lafan")
    p.add_argument("--motion", default="walk1_subject1", help="CSV stem or filename")
    p.add_argument("--list_motions", action="store_true")
    p.add_argument("--seconds", type=float, help="limit simulated duration; default: full motion")
    p.add_argument("--start", type=float, default=0.0, help="reference start time in seconds")
    p.add_argument("--headless", action="store_true", help="run as fast as possible without a viewer")
    p.add_argument("--video", type=Path, help="write a 25 fps MP4; also enables headless mode")
    p.add_argument("--report", type=Path, help="write metrics as JSON")
    p.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    p.add_argument("--substeps", type=int, default=4, help="physics steps per 50 Hz policy action")
    p.add_argument("--no_reference", action="store_true", help="hide the reference skeleton")
    p.add_argument("--continue_after_failure", action="store_true", help="finish the requested interval and visibly mark any tracking termination")
    p.add_argument("--clip_label", default="", help="optional segment label in video")
    a = p.parse_args()
    if a.start < 0 or not math.isfinite(a.start):
        p.error("--start must be finite and nonnegative")
    if a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds < 0.04):
        p.error("--seconds must be at least 0.04")
    if a.threads < 1 or a.substeps < 1:
        p.error("--threads and --substeps must be positive")
    return a


def resolve_checkpoint(checkpoint=None, run_dir=None):
    """Select the highest numbered snapshot, or respect an explicit selection."""
    if checkpoint is not None:
        return Path(checkpoint)
    directories = (Path(run_dir),) if run_dir is not None else (REPO/"logs/e1_paper", REPO/"models")
    candidates = []
    for priority, directory in enumerate(directories):
        for path in directory.glob("ckpt_*.pt"):
            match = re.fullmatch(r"ckpt_(\d+)\.pt", path.name)
            if match and path.is_file():
                candidates.append((int(match[1]), -priority, path))
    if not candidates:
        searched = ", ".join(str(p) for p in directories)
        raise FileNotFoundError(f"No numbered checkpoints in {searched}; use --checkpoint")
    return max(candidates, key=lambda item: item[:2])[2]


def build_model(mjcf, joint_names, drives, substeps):
    """Build a separate in-memory model with the training PD drives and contacts."""
    import mujoco
    import numpy as np

    src = Path(mjcf)
    root = ET.parse(src).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((src.parent / compiler.get("meshdir", ".")).resolve()))
    # Original asset motors/sensors/keyframes do not describe the trained drives.
    for tag in ("actuator", "sensor", "keyframe", "contact"):
        for element in root.findall(tag):
            root.remove(element)
    actuators = ET.SubElement(root, "actuator")
    for name in joint_names:
        kp, kd, effort, _, _ = drives(name)
        ET.SubElement(actuators, "position", name=f"policy_{name}", joint=name,
                      kp=str(kp), kv=str(kd), ctrllimited="false",
                      forcelimited="true", forcerange=f"{-effort} {effort}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = 0.02 / substeps
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.tolerance = 1e-6
    ids = np.array([model.joint(n).id for n in joint_names])
    if model.nq != 7 + len(ids) or model.nv != 6 + len(ids):
        raise ValueError("Expected one floating root and exactly the policy's hinge joints")
    dofs = model.jnt_dofadr[ids]
    # Kd is integrated implicitly in each position servo, not added twice as
    # passive joint damping. Preserve the asset's joint friction and geometry.
    model.dof_damping[dofs] = 0.0
    model.dof_armature[dofs] = [drives(n)[4] for n in joint_names]
    # Match the training environment's disabled robot self collisions.
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot = collision & (model.geom_bodyid != 0)
    ground = collision & (model.geom_bodyid == 0)
    model.geom_contype[robot], model.geom_conaffinity[robot] = 2, 1
    model.geom_contype[ground], model.geom_conaffinity[ground] = 1, 2
    model.geom_friction[collision, 0] = 1.0
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    return model


class NativeRobot:
    """Named joint mapping and link-origin/world-frame state conventions."""

    def __init__(self, model, names, drives):
        import mujoco
        import numpy as np

        self.model, self.data = model, mujoco.MjData(model)
        self.root = model.body("pelvis").id
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.jnt_bodyid[free[0]] != self.root:
            raise ValueError("The policy expects the pelvis to be the floating root")
        self.root_q, self.root_v = int(model.jnt_qposadr[free[0]]), int(model.jnt_dofadr[free[0]])
        ids = np.array([model.joint(n).id for n in names])
        self.qadr, self.vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
        self.act = np.array([model.actuator(f"policy_{n}").id for n in names])
        self.scale = np.array([0.25 * drives(n)[2] / drives(n)[0] for n in names])

    def reset(self, state):
        import mujoco
        from supermimic.rotations import mat_to_quat

        p, R, v, w, q, qd = [s[0] for s in state]
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.root_q:self.root_q + 3] = p.numpy()
        d.qpos[self.root_q + 3:self.root_q + 7] = mat_to_quat(R).numpy()  # wxyz
        d.qpos[self.qadr] = q.numpy()
        d.qvel[self.root_v:self.root_v + 3] = v.numpy()
        d.qvel[self.root_v + 3:self.root_v + 6] = (R.T @ w).numpy()  # local angular velocity
        d.qvel[self.vadr] = qd.numpy()
        d.ctrl[self.act] = q.numpy()
        mujoco.mj_forward(self.model, d)

    def state(self):
        import mujoco
        import numpy as np
        import torch

        d = self.data
        vel = np.empty(6)
        # XBODY is the link origin; BODY would return the inertial frame/CoM.
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, self.root, vel, 0)
        arrays = (d.xpos[self.root], d.xmat[self.root].reshape(3, 3), vel[3:], vel[:3],
                  d.qpos[self.qadr], d.qvel[self.vadr])
        return tuple(torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in arrays)

    def step(self, action, default_q, substeps):
        import mujoco
        import numpy as np

        self.data.ctrl[self.act] = default_q + self.scale * action
        mujoco.mj_step(self.model, self.data, nstep=substeps)
        # mj_step integrates qpos last; refresh derived poses/velocities before
        # observations, rewards, or rendering at the next control boundary.
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo state became nonfinite")


def reference_skeleton(scene, positions, clear=True):
    import mujoco
    import numpy as np

    if clear:
        scene.ngeom = 0
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
             (7, 8), (8, 9), (9, 10), (7, 11), (11, 12), (12, 13)]
    for a, b in edges:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), np.array([0.1, 0.65, 1.0, 0.65]))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, positions[a], positions[b])
        scene.ngeom += 1


def run(args):
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        args.headless = True
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for --video")
    os.environ["SUPERMIMIC_ROBOT"] = "e1"
    sys.path.insert(0, str(REPO))
    import mujoco
    import numpy as np
    import torch

    from supermimic.kinematics import RobotKinematics
    from supermimic.mdp import MDP, OBS_DIM, MAX_STEP_REWARD, XI_DIM
    from supermimic.motion import CONTROL_DT, GOAL_FRAME_DIM, MotionLib
    from supermimic.networks import Policy
    from supermimic.robot import JOINT_NAMES, MJCF_PATH, default_joint_pos_list, drive_parameters

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    checkpoint = resolve_checkpoint(args.checkpoint, args.run_dir)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ck["args"]
    if cfg.get("robot") != "e1":
        raise ValueError("This sim2sim runner requires an E1 checkpoint")
    H, K = int(cfg["hist"]), int(cfg["goal_frames"])
    policy = Policy(OBS_DIM, GOAL_FRAME_DIM * K, hist=H).eval()
    policy.load_state_dict(ck["pi"])
    iteration = int(ck["iteration"])
    del ck
    src = args.motion_dir / (Path(args.motion).stem + ".csv")
    if not src.is_file():
        raise FileNotFoundError(f"Motion not found: {src}; use --list_motions")
    kin = RobotKinematics("cpu")
    motion = MotionLib([str(src)], kin, "cpu")
    default_q = torch.tensor(default_joint_pos_list())
    mdp = MDP(kin, motion, default_q)
    model = build_model(MJCF_PATH, JOINT_NAMES, drive_parameters, args.substeps)
    robot = NativeRobot(model, JOINT_NAMES, drive_parameters)
    start = int(round(args.start / CONTROL_DT))
    clip_end = int(motion.clip_end[0])
    if start >= clip_end:
        raise ValueError(f"--start exceeds motion duration {clip_end * CONTROL_DT:.2f}s")
    last = clip_end if args.seconds is None else min(clip_end, start + int(round(args.seconds / CONTROL_DT)))
    xi = torch.zeros(1, XI_DIM)
    xi[:, 0] = 1.0
    events = deque()
    paused = False
    f, hist, prev = torch.tensor([start]), None, torch.zeros(1, len(JOINT_NAMES))
    robot.reset(motion.ref_state(f))
    total, count, failed, stopped = 0.0, 0, False, False
    first_termination = None
    pos_error, torque_peak = [], 0.0
    wall = time.monotonic()
    print(f"E1 sim2sim | native MuJoCo {mujoco.__version__} | checkpoint {iteration} | {src.stem}", flush=True)
    print("Policy 50 Hz, physics %.0f Hz, FP32 CPU inference. Viewer: Space pause, R restart, Esc close." % (1 / model.opt.timestep), flush=True)

    with ExitStack() as stack, torch.inference_mode():
        viewer, renderer, video = None, None, None
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.0, 135.0, -15.0
        if not args.headless:
            import mujoco.viewer
            viewer = stack.enter_context(mujoco.viewer.launch_passive(model, robot.data, key_callback=events.append))
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 135.0, -15.0
                viewer.opt.geomgroup[3] = 0  # hide collision shapes, keep robot visual meshes
        if args.video:
            from PIL import Image, ImageDraw, ImageFont
            args.video.parent.mkdir(parents=True, exist_ok=True)
            renderer = stack.enter_context(mujoco.Renderer(model, height=540, width=960))
            vis = mujoco.MjvOption()
            vis.geomgroup[3] = 0
            video = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-s", "960x540", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
                                      "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(args.video)],
                                     stdin=subprocess.PIPE)
            font = ImageFont.load_default(size=18)
        try:
            while int(f[0]) < last:
                tick = time.monotonic()
                just_failed = False
                if viewer and not viewer.is_running():
                    stopped = True
                    break
                while events:
                    key = events.popleft()
                    if key == 32:
                        paused = not paused
                    elif key in (ord("R"), ord("r")):
                        f, hist, prev = torch.tensor([start]), None, torch.zeros_like(prev)
                        robot.reset(motion.ref_state(f))
                        total, count, pos_error, torque_peak = 0.0, 0, [], 0.0
                        failed, first_termination = False, None
                if not paused:
                    S = robot.state()
                    o = mdp.obs(S, prev, xi, f)
                    hist = o.unsqueeze(1).repeat(1, H, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
                    action = policy(hist, motion.goal(f, K))
                    if not torch.isfinite(action).all():
                        raise FloatingPointError("Policy returned a nonfinite action")
                    robot.step(action[0].numpy(), default_q.numpy(), args.substeps)
                    f += 1
                    S1 = robot.state()
                    reward, terms = mdp.reward(S1, f, action, prev, terms=True)
                    total += float(reward[0])
                    pos_error.append(math.sqrt(float(terms["e_body_pos"][0])))
                    torque_peak = max(torque_peak, float(np.abs(robot.data.actuator_force).max()))
                    count += 1
                    prev = action
                    terminated = bool(mdp.termination(S1, f)[0])
                    just_failed = terminated and not failed
                    if just_failed:
                        first_termination = int(f[0]) * CONTROL_DT
                    failed = failed or terminated
                positions = motion.body_p[int(f[0])].numpy()
                if viewer:
                    with viewer.lock():
                        viewer.cam.lookat[:] = robot.data.xpos[robot.root] + np.array([0, 0, 0.15])
                        if not args.no_reference:
                            reference_skeleton(viewer.user_scn, positions)
                    viewer.sync()
                if video and count % 2 == 0:
                    camera.lookat[:] = robot.data.xpos[robot.root] + np.array([0, 0, 0.15])
                    renderer.update_scene(robot.data, camera=camera, scene_option=vis)
                    if not args.no_reference:
                        reference_skeleton(renderer.scene, positions, clear=False)
                    img = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(img)
                    label = f"E1 | Isaac Lab -> native MuJoCo | checkpoint {iteration:,}"
                    draw.rectangle((0, 0, 960, 60), fill=(245, 245, 245))
                    draw.text((12, 7), label, fill=(20, 20, 20), font=font)
                    prefix = getattr(args, "clip_label", "")
                    prefix = prefix + " | " if prefix else ""
                    draw.text((12, 33), f"{prefix}{src.stem} | t={int(f[0])*CONTROL_DT:.2f}s | blue: reference", fill=(20, 20, 20), font=font)
                    if failed:
                        draw.rectangle((0, 500, 960, 540), fill=(145, 20, 20))
                        draw.text((12, 510), f"Tracking termination at reference t={first_termination:.2f}s; continuing selected segment", fill="white", font=font)
                    video.stdin.write(np.asarray(img).tobytes())
                if just_failed:
                    print(f"Tracking termination at reference t={int(f[0]) * CONTROL_DT:.2f}s", flush=True)
                if failed and not getattr(args, "continue_after_failure", False):
                    break
                if viewer:
                    time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - tick)))
                elif count and count % 500 == 0:
                    print(f"Simulated {count * CONTROL_DT:.1f}s / {(last-start) * CONTROL_DT:.1f}s", flush=True)
        finally:
            if video:
                video.stdin.close()
                if video.wait() != 0:
                    raise RuntimeError("ffmpeg failed to encode the MP4")
    requested = last - start
    result = {"backend": "native_mujoco", "checkpoint": str(checkpoint.resolve()), "iteration": iteration,
              "motion": src.stem, "start_seconds": start * CONTROL_DT, "sim_seconds": count * CONTROL_DT,
              "motion_duration_seconds": clip_end * CONTROL_DT, "requested_seconds": requested * CONTROL_DT,
              "failed": failed, "stopped_by_user": stopped,
              "first_termination_seconds": first_termination,
              "completed_motion": not failed and not stopped and int(f[0]) >= clip_end,
              "completed_requested_duration": not failed and not stopped and count >= requested,
              "simulated_requested_duration": not stopped and count >= requested,
              "norm_return_requested": total / (requested * MAX_STEP_REWARD),
              "mean_body_position_error_m": float(np.mean(pos_error)) if pos_error else None,
              "peak_actuator_torque_nm": torque_peak, "wall_seconds": time.monotonic() - wall,
              "video": str(args.video.resolve()) if args.video else None}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    opts = arguments()
    if opts.list_motions:
        print("\n".join(p.stem for p in sorted(opts.motion_dir.glob("*.csv"))))
    else:
        run(opts)
