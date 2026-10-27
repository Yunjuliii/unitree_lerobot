"""Plot dataset arm joint targets against measured joint angles from replay runs.

Usage:
    python plot_replay_tracking.py --dataset-dir <lerobot_dir> --episode 0 \
        --run "无重力补偿=gc_off.npz" --run "有重力补偿=gc_on.npz" --out tracking.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

LABELS = [
    "L shoulder pitch", "L shoulder roll", "L shoulder yaw", "L elbow", "L wrist roll", "L wrist pitch", "L wrist yaw",
    "R shoulder pitch", "R shoulder roll", "R shoulder yaw", "R elbow", "R wrist roll", "R wrist pitch", "R wrist yaw",
]
TRUTH_COLOR = "#1a1a19"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"
TEXT_SECONDARY = "#5f5e58"


def load_dataset_arm(dataset_dir, episode):
    with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
        info = json.load(f)
    chunk = episode // int(info.get("chunks_size", 1000))
    path = os.path.join(dataset_dir, info["data_path"].format(episode_chunk=chunk, episode_index=episode))
    t = pq.read_table(path, columns=["action", "frame_index"])
    order = np.argsort(np.asarray(t.column("frame_index").to_pylist()))
    return np.stack(t.column("action").to_numpy(zero_copy_only=False))[order][:, :14], float(info["fps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=NPZ")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    truth, fps = load_dataset_arm(os.path.expanduser(args.dataset_dir), args.episode)
    runs = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        d = np.load(os.path.expanduser(path))
        runs.append((label, d["q_meas"]))
    T = min([len(truth)] + [len(q) for _, q in runs])
    t = np.arange(T) / fps

    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"], "font.size": 9})
    fig, axes = plt.subplots(7, 2, figsize=(13, 15), sharex=True, facecolor=SURFACE)
    for j in range(14):
        ax = axes[j % 7, j // 7]
        ax.set_facecolor(SURFACE)
        ax.plot(t, truth[:T, j], color=TRUTH_COLOR, lw=1.4, ls="--", label="数据集目标值 (action)")
        for k, (label, q) in enumerate(runs):
            ax.plot(t, q[:T, j], color=SERIES_COLORS[k], lw=1.8, label=label)
        ax.set_title(LABELS[j], loc="left", fontsize=10, color=TRUTH_COLOR, pad=3)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        if j % 7 == 6:
            ax.set_xlabel("时间 (s)", color=TEXT_SECONDARY)
        if j // 7 == 0:
            ax.set_ylabel("关节角 (rad)", color=TEXT_SECONDARY)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"Episode {args.episode} 回放：数据集目标 vs 实测关节角", y=0.998, fontsize=13, color=TRUTH_COLOR)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
