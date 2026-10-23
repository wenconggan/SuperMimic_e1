# Recorded simulation results

This directory contains snapshots of recorded E1 experiments. Local absolute paths were converted to relative artifact paths; numerical measurements were retained. `provenance.json` records original-source and distributed-file SHA-256 hashes. Historical checkpoint names in reports identify the evaluated artifact. The local prepared folder includes the reference checkpoint in `models/`; checkpoint files are excluded from Git commits.

| Checkpoint | Protocol | Completion | Mean normalized tracking return |
| --- | --- | --- | --- |
| 14,000 | 40 complete training-library motions in Isaac Lab / Newton | 39/40 (97.5%) | 0.7984306 |
| 32,000 | Same full-motion protocol | 39/40 (97.5%) | 0.8142619 |
| 164,000 | Same full-motion protocol | 40/40 (100%) | 0.8558580 |
| 168,000 | 10 randomly selected, independent 8-second intervals in native MuJoCo | 10/10 intervals (100%) | See per-interval `norm_return_requested` |

The full-motion evaluation runs each motion from its first frame until its end or a height/tilt termination, with domain randomization and observation/action noise disabled and friction 1.0. These motions belong to the training library; this is not a held-out generalization benchmark.

The MuJoCo montage uses seed **20261023**, the exact motion names and start frames in `random10_sampling_plan.json`, 50 Hz control, and a 25 fps recording. Each interval starts independently at its reference state. Tracking termination would be latched, marked in red, and kept in the video rather than rejected. All 10 intervals completed their requested duration without termination. `completed_motion: false` in the interval reports means the selected excerpt does not span the entire source motion. The source segments were concatenated into `../assets/e1_random10_80s.mp4`; `montage_time_range_seconds` identifies each interval in that file.

`train_log_168000.csv` is a fixed training-log snapshot ending at iteration 168000. Its `sim_rew` column measures mean tracking reward during training; `term_frac` measures rollout termination fraction and is not the full-motion failure rate. In the figure, training curves show the raw samples and a trailing mean over 100 logged samples (2000 iterations). Full-motion evaluation is plotted only at the three recorded checkpoints.

The motion plan includes source CSV SHA-256 hashes and the evaluated checkpoint SHA-256. Media hashes, frame extraction times, and GIF cut intervals are recorded in `../assets/media_manifest.json`.
