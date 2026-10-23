"""Record seeded random LaFAN1 intervals under the E1 policy, then concatenate MP4s."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
from types import SimpleNamespace

from sim2sim import REPO, run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_frame(rng, maximum, length, previous):
    """Uniform draw from all integer starts whose intervals do not overlap earlier draws."""
    spans = [(0, maximum)]
    for used in previous:
        forbidden_lo, forbidden_hi = used-length+1, used+length-1
        remaining = []
        for lo, hi in spans:
            if forbidden_hi < lo or forbidden_lo > hi:
                remaining.append((lo, hi))
            else:
                if lo < forbidden_lo:
                    remaining.append((lo, forbidden_lo-1))
                if hi > forbidden_hi:
                    remaining.append((forbidden_hi+1, hi))
        spans = remaining
    count = sum(hi-lo+1 for lo, hi in spans)
    if count <= 0:
        raise ValueError("Not enough non-overlapping intervals; reduce sample count")
    offset = rng.randrange(count)
    for lo, hi in spans:
        size = hi-lo+1
        if offset < size:
            return lo+offset
        offset -= size
    raise AssertionError("Unreachable interval offset")


def make_plan(args):
    rng = random.Random(args.seed)
    library, groups = {}, defaultdict(list)
    length = round(args.seconds/.02)
    for path in sorted(args.motion_dir.glob("*.csv")):
        rows = sum(1 for line in path.open() if line.strip())
        # Same 30 -> 50 Hz frame count as MotionLib._load_csv.
        end = math.floor(((rows-1)/30.0)/.02)
        if end < length:
            raise ValueError(f"{path.name} is shorter than {args.seconds}s")
        category = re.sub(r"\d+$", "", path.stem.split("_subject")[0])
        library[path.stem] = {"category": category, "source_rows": rows,
                              "end_frame": end, "sha256": digest(path)}
        groups[category].append(path.stem)
    if not library:
        raise ValueError(f"No CSV motions in {args.motion_dir}")
    choices = []
    if args.scope == "total":
        if args.samples > len(library):
            raise ValueError("Total sampling requires distinct motion files")
        choices = rng.sample(sorted(library), args.samples)
    else:
        for category in sorted(groups):
            category_choices = []
            while len(category_choices) < args.samples:
                cycle = groups[category].copy()
                rng.shuffle(cycle)
                category_choices.extend(cycle[:args.samples-len(category_choices)])
            choices.extend(category_choices)
    used, indices, clips = defaultdict(list), Counter(), []
    for index, name in enumerate(choices, 1):
        info = library[name]
        start = start_frame(rng, info["end_frame"]-length, length, used[name])
        used[name].append(start)
        indices[info["category"]] += 1
        clips.append({"index": index, "category": info["category"],
                      "category_index": indices[info["category"]], "motion": name,
                      "start_frame": start, "end_frame": start+length,
                      "start_seconds": round(start*.02, 8), "seconds": args.seconds})
    return {"format": "e1_lafan_random_montage_v1", "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": digest(args.checkpoint), "seed": args.seed,
            "scope": args.scope, "samples": args.samples, "seconds_per_clip": args.seconds,
            "control_hz": 50, "video_fps": 25, "width": 960, "height": 540,
            "motion_dir": str(args.motion_dir.resolve()), "library": library, "clips": clips,
            "protocol": "Each seeded interval starts independently from its reference state. "
                        "A trained policy drives native MuJoCo, with no added perturbations. "
                        "Tracking terminations stay in the video, latched and marked in red; "
                        "simulation continues to the end of each selected interval. No rejection of failed trials."}


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=codec_name,width,height,nb_frames,avg_frame_rate",
                             "-show_entries", "format=duration,size", "-of", "json", str(path)],
                            check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def validate_video(path, seconds):
    info = probe(path)
    stream = info["streams"][0]
    assert stream["codec_name"] == "h264" and (stream["width"], stream["height"]) == (960, 540)
    assert stream["avg_frame_rate"] == "25/1"
    assert int(stream["nb_frames"]) == round(seconds*25)
    assert abs(float(info["format"]["duration"])-seconds) < .05
    return info


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion_dir", type=Path, default=REPO/"datasets"/"e1_lafan")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("total", "per-category"), default="total")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--plan_only", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or not math.isfinite(args.seconds) or args.seconds < .04 or not math.isclose(args.seconds*25, round(args.seconds*25)):
        parser.error("samples must be positive; seconds must be a positive multiple of 1/25")
    if not args.checkpoint.is_file():
        parser.error("checkpoint must be a saved file")
    args.checkpoint, args.motion_dir, args.output_dir = (p.resolve() for p in (args.checkpoint, args.motion_dir, args.output_dir))
    plan = make_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir/"sampling_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Output directory contains a different sampling plan; choose a new directory")
    write_json(plan_path, plan)
    counts = Counter(c["category"] for c in plan["clips"])
    print(f"Frozen plan: {len(plan['clips'])} clips x {args.seconds}s; categories={dict(counts)}; checkpoint={args.checkpoint.name}", flush=True)
    if args.plan_only:
        return
    started, reports = time.monotonic(), []
    clip_dir = args.output_dir/"clips"
    clip_dir.mkdir(exist_ok=True)
    summary_path = args.output_dir/"summary.json"
    for clip in plan["clips"]:
        stem = f"{clip['index']:03d}_{clip['motion']}_{clip['start_frame']:06d}"
        video, report, log = (clip_dir/(stem+suffix) for suffix in (".mp4", ".json", ".log"))
        if video.exists() and report.exists():
            result = json.loads(report.read_text())
            validate_video(video, args.seconds)
            assert result["checkpoint"] == str(args.checkpoint) and result["start_seconds"] == clip["start_seconds"]
            assert result["simulated_requested_duration"]
            print(f"Reuse {clip['index']}/{len(plan['clips'])}: {clip['motion']}", flush=True)
        else:
            print(f"Simulate {clip['index']}/{len(plan['clips'])}: {clip['category']} | {clip['motion']} | {clip['start_seconds']:.2f}-{clip['start_seconds']+args.seconds:.2f}s", flush=True)
            options = SimpleNamespace(run_dir=args.checkpoint.parent, checkpoint=args.checkpoint,
                                      motion_dir=args.motion_dir, motion=clip["motion"], seconds=args.seconds,
                                      start=clip["start_seconds"], headless=True, video=video, report=report,
                                      threads=2, substeps=4, no_reference=False, continue_after_failure=True,
                                      clip_label=f"{clip['index']:02d}/{len(plan['clips'])}" +
                                                 (f" {clip['category_index']:02d}/{args.samples}" if args.scope == "per-category" else ""))
            with log.open("w") as file, redirect_stdout(file), redirect_stderr(file):
                result = run(options)
            validate_video(video, args.seconds)
            assert result["simulated_requested_duration"], f"Incomplete simulation: {stem}"
        reports.append({**clip, **result, "segment_video": str(video)})
        failed = sum(bool(r["failed"]) for r in reports)
        write_json(summary_path, {"plan": str(plan_path), "clips_finished": len(reports),
                                 "clips_planned": len(plan["clips"]), "failed": failed,
                                 "results": reports, "complete": False})
        print(f"Recorded {len(reports)}/{len(plan['clips'])}; tracking terminations={failed}", flush=True)
    concat = args.output_dir/"concat.txt"
    def quote(path):
        return str(path).replace("'", "'\\''")
    concat.write_text("".join(f"file '{quote(r['segment_video'])}'\n" for r in reports))
    output = args.output_dir/"lafan1_random_8s_montage.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", "-movflags", "+faststart", str(output)], check=True)
    info = validate_video(output, len(reports)*args.seconds)
    failed = sum(bool(r["failed"]) for r in reports)
    summary = {"plan": str(plan_path), "checkpoint": str(args.checkpoint),
               "clips_planned": len(reports), "clips_finished": len(reports),
               "seconds_per_clip": args.seconds, "video_seconds": len(reports)*args.seconds,
               "tracking_completed_without_termination": len(reports)-failed,
               "tracking_terminated": failed, "completion_rate": (len(reports)-failed)/len(reports),
               "complete": True, "video": str(output), "video_probe": info,
               "wall_seconds": time.monotonic()-started, "results": reports}
    write_json(summary_path, summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
