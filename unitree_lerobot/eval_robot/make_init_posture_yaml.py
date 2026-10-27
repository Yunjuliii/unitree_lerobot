#!/usr/bin/env python3
"""
Write the average initial posture YAML that ``--init-state-yaml`` of eval_h2_groot*.py reads, from a
LeRobot dataset whose ``observation.state`` holds the H2 arm joints + Sharpa joints
(58-dim: 14 arm joints in H2_JointArmIndex order, then 22 left-hand + 22 right-hand joints, rad),
e.g. ``trocar_plate_lerobot_0915`` or ``trocar_test_lerobot``.

The eef datasets (``*_eef_camright_*``, 62-dim) do not contain the arm joints; use the joint-space
recording they were built from (see their README.md).

Layout (= teleop/utils/compute_average_initial_state.py):
    first_n_frames, num_episodes, num_frames, source_dataset,
    states: {left_arm, right_arm: qpos[7]; left_ee, right_ee: qpos[22]}

Usage (tv env):
    python make_init_posture_yaml.py ~/isaac-ego-inpaint/dataset/trocar_plate_lerobot_0915 \
        --output init_post_trocar_0915.yaml --frame 25  # mean over episodes of the frame with frame_index 25
    python make_init_posture_yaml.py <dataset> --output <yaml> --first-n 3   # mean of the first 3 frames of every episode
    python make_init_posture_yaml.py <dataset> --output <yaml> --frame 25 --compare init_post_trocar.yaml
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ARM_DOF = 14
HAND_DOF = 22
STATE_DIM = ARM_DOF + 2 * HAND_DOF  # 58


STATE_COLUMNS = ("observation.state", "joints")  # 58-dim joint column, by dataset generation


def load_frames(dataset_dir: str, *, first_n: int | None = None, frame: int | None = None,
                column: str | None = None) -> tuple[np.ndarray, int, str]:
    """(frames, n_episodes, column): 58-dim joint rows with frame_index < first_n, or with frame_index == frame,
    LeRobot v2.1 (data/chunk-*/episode_*.parquet) or v3 (data/chunk-*/file-*.parquet) layout.  The joint column is
    ``observation.state`` (trocar_test_lerobot, trocar_plate_lerobot_0915) or ``joints`` (trocar_tele_0921 and later)
    unless ``column`` is given."""
    if (first_n is None) == (frame is None):
        raise ValueError("give exactly one of first_n / frame")
    files = sorted(glob.glob(os.path.join(dataset_dir, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {os.path.join(dataset_dir, 'data')}")
    if column is None:
        names = pq.read_schema(files[0]).names
        column = next((c for c in STATE_COLUMNS if c in names), None)
        if column is None:
            raise KeyError(f"no joint column {STATE_COLUMNS} in {files[0]} (columns: {names}); pass --state-column")
    rows, episodes = [], set()
    for f in files:
        table = pq.read_table(f, columns=[column, "frame_index", "episode_index"])
        state = np.stack(table.column(column).to_numpy(zero_copy_only=False)).astype(np.float64)
        frame_index = table.column("frame_index").to_numpy()
        episode_index = table.column("episode_index").to_numpy()
        mask = (frame_index == frame) if frame is not None else (frame_index < first_n)
        rows.append(state[mask])
        episodes.update(np.unique(episode_index[mask]).tolist())
    frames = np.concatenate(rows, axis=0)
    if frames.shape[1] != STATE_DIM:
        raise ValueError(
            f"{column} is {frames.shape[1]}-dim, expected {STATE_DIM} (14 arm + 22 + 22 hand joints). "
            "An eef dataset (62-dim) has no arm joints: use the joint-space recording it was built from."
        )
    return frames, len(episodes), column


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="LeRobot dataset dir with 58-dim observation.state")
    ap.add_argument("--output", required=True, help="YAML to write")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--first-n", type=int, default=None, metavar="N",
                   help="average the first N frames of every episode (default when --frame is not given: 3)")
    g.add_argument("--frame", type=int, default=None, metavar="IDX",
                   help="average the single frame with frame_index == IDX over all episodes (0-based; episodes shorter "
                        "than IDX+1 frames are skipped)")
    ap.add_argument("--compare", default=None, metavar="YAML", help="print the joint differences to this init YAML")
    ap.add_argument("--state-column", default=None, metavar="COL",
                    help="parquet column with the 58 joints (default: observation.state, else joints)")
    args = ap.parse_args()

    import yaml

    root = os.path.abspath(os.path.expanduser(args.dataset))
    if args.frame is not None:
        frames, n_episodes, column = load_frames(root, frame=args.frame, column=args.state_column)
        selection = {"frame_index": int(args.frame)}
        desc = f"frame_index {args.frame} of every episode"
    else:
        first_n = 3 if args.first_n is None else args.first_n
        frames, n_episodes, column = load_frames(root, first_n=first_n, column=args.state_column)
        selection = {"first_n_frames": int(first_n)}
        desc = f"first {first_n} frames of every episode"
    if frames.shape[0] == 0:
        raise SystemExit(f"no frames selected ({desc}) in {root}")
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    payload = {
        **selection,
        "num_episodes": int(n_episodes),
        "num_frames": int(frames.shape[0]),
        "source_dataset": f"{os.path.basename(os.path.dirname(root))}/{os.path.basename(root)} "
                          f"({column}, {STATE_DIM}-dim; mean of {desc})",
        "states": {
            "left_arm": {"qpos": mean[0:7].tolist()},
            "right_arm": {"qpos": mean[7:14].tolist()},
            "left_ee": {"qpos": mean[14:36].tolist()},
            "right_ee": {"qpos": mean[36:58].tolist()},
        },
    }
    out = os.path.expanduser(args.output)
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print(f"wrote {out}: mean of {desc} -> {frames.shape[0]} frames from {n_episodes} episodes of {root}")
    print(f"  left_arm  (rad) {mean[0:7]}   spread std (deg) {np.degrees(std[0:7])}")
    print(f"  right_arm (rad) {mean[7:14]}   spread std (deg) {np.degrees(std[7:14])}")
    print(f"  left_ee / right_ee: 22 joints each, mean |q| {np.abs(mean[14:36]).mean():.3f} / {np.abs(mean[36:58]).mean():.3f} rad")
    if args.compare:
        other = yaml.safe_load(open(os.path.expanduser(args.compare)))["states"]
        other_arm = np.array(other["left_arm"]["qpos"] + other["right_arm"]["qpos"], dtype=np.float64)
        other_hand = np.array(other["left_ee"]["qpos"] + other["right_ee"]["qpos"], dtype=np.float64)
        print(f"  vs {args.compare}: arm diff (deg) {np.degrees(mean[:14] - other_arm)}")
        print(f"  vs {args.compare}: max |hand diff| {np.degrees(np.abs(mean[14:] - other_hand)).max():.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
