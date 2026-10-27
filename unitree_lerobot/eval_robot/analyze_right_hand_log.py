"""Per-joint right-hand offset: real-robot teleop grasp vs. the model's live output.

Reference = the REAL-ROBOT teleop episodes of the training dataset (their
observation.state is the raw H2+Sharpa recording): grasp phase of each episode
= top --top-frac of mean flexion over thumb_MCP_FE, index_MCP_FE, index_PIP,
middle_MCP_FE, middle_PIP; the reference pose is the median (plus q25/q75) of
the right-hand joints (dims 36-57) over those frames.

Live = the ep*_hand_log.npz written by eval_h2_groot.py --hand-log-dir; the
grasp phase is found with the same rule (or --grasp-window), and

    needed_offset = tele_grasp_median - model_grasp_median

is what to ADD to the absolute right-hand target the client receives from the
server (relative deltas already added back). Do not modify the state fed to
the model.

Usage:
    python analyze_right_hand_log.py --log-dir ~/hand_logs/20260916_115124
    python analyze_right_hand_log.py --log-dir ~/hand_logs/20260916_115124 --grasp-window 3.8 6.1
    python analyze_right_hand_log.py --log-dir ... --source cmd   # after blend/EMA instead of raw server output
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

DEFAULT_DATASET = "/home/yunju/isaac-ego-inpaint/dataset/trocar_cotrain_joints_0915"
RIGHT_HAND_SLICE = slice(36, 58)
JOINT_NAMES = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
]
FLEXION_IDX = [JOINT_NAMES.index(n) for n in
               ["right_thumb_MCP_FE", "right_index_MCP_FE", "right_index_PIP",
                "right_middle_MCP_FE", "right_middle_PIP"]]


def grasp_mask(hand, top_frac, window=None, t_rel=None):
    if window is not None:
        return (t_rel >= window[0]) & (t_rel <= window[1])
    flex = np.nanmean(hand[:, FLEXION_IDX], axis=1)
    return flex >= np.nanquantile(flex, 1.0 - top_frac)


def dataset_grasp_reference(dataset_dir, episodes, top_frac):
    rows = []
    for ep in episodes:
        path = os.path.join(dataset_dir, "data", "chunk-000", f"episode_{ep:06d}.parquet")
        table = pq.read_table(path, columns=["observation.state"])
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
        hand = state[:, RIGHT_HAND_SLICE]
        rows.append(hand[grasp_mask(hand, top_frac)])
    allrows = np.concatenate(rows)
    return (np.median(allrows, axis=0), np.quantile(allrows, 0.25, axis=0),
            np.quantile(allrows, 0.75, axis=0), len(rows), len(allrows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True, help="Run directory with ep*_hand_log.npz (or a single .npz)")
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET,
                    help="LeRobot v2.1 dataset whose real-robot teleop episodes give the reference grasp")
    ap.add_argument("--tele-episodes", type=int, nargs=2, default=(0, 54), metavar=("FIRST", "LAST"),
                    help="Inclusive episode_index range of the REAL-ROBOT teleop episodes (default 0 54)")
    ap.add_argument("--source", choices=["raw", "cmd", "meas"], default="raw",
                    help="raw = server output (absolute), cmd = after blend/EMA, meas = measured joints")
    ap.add_argument("--top-frac", type=float, default=0.2, help="Grasp phase = top fraction of flexion")
    ap.add_argument("--grasp-window", type=float, nargs=2, metavar=("START_S", "END_S"),
                    help="Live grasp phase = this window (seconds since the episode's first tick) instead of the flexion rule")
    ap.add_argument("--out", default=None, help="Output JSON (default: <log-dir>/right_hand_offset_from_live.json)")
    args = ap.parse_args()

    episodes = range(args.tele_episodes[0], args.tele_episodes[1] + 1)
    tele, q25, q75, n_eps, n_frames = dataset_grasp_reference(args.dataset_dir, episodes, args.top_frac)
    print(f"real-robot reference: {n_eps} teleop episodes, {n_frames} grasp frames from {args.dataset_dir}")

    paths = ([args.log_dir] if args.log_dir.endswith(".npz")
             else sorted(glob.glob(os.path.join(os.path.expanduser(args.log_dir), "ep*_hand_log.npz"))))
    if not paths:
        raise SystemExit(f"no ep*_hand_log.npz under {args.log_dir}")

    key = {"raw": "right_hand_raw", "cmd": "right_hand_cmd", "meas": "right_hand_meas"}[args.source]
    per_episode, pooled = {}, []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if key not in d.files or len(d[key]) == 0:
            print(f"[skip] {os.path.basename(p)}: no {key}")
            continue
        hand = d[key].astype(np.float64)
        ok = ~np.isnan(hand).any(axis=1)
        hand = hand[ok]
        t_rel = d["wall_time"][ok] - d["wall_time"][ok][0]
        mask = grasp_mask(hand, args.top_frac, args.grasp_window, t_rel)
        if mask.sum() == 0:
            print(f"[skip] {os.path.basename(p)}: empty grasp phase")
            continue
        per_episode[os.path.basename(p)] = {
            "n_ticks": int(len(hand)), "n_grasp_ticks": int(mask.sum()),
            "grasp_t_rel_s": [float(t_rel[mask].min()), float(t_rel[mask].max())],
            "model_grasp_median": np.median(hand[mask], axis=0).tolist(),
        }
        pooled.append(hand[mask])
        print(f"{os.path.basename(p)}: {len(hand)} ticks, grasp phase {mask.sum()} ticks "
              f"@ {t_rel[mask].min():.1f}-{t_rel[mask].max():.1f}s")
    if not pooled:
        raise SystemExit("no usable episodes")

    model = np.median(np.concatenate(pooled), axis=0)
    needed = tele - model
    outside = (model < q25) | (model > q75)

    print(f"\nsource={args.source}  ({len(pooled)} live episodes pooled)   needed = tele_median - model")
    print(f"{'joint':24s} {'tele_med':>8s} {'q25':>7s} {'q75':>7s} {'model':>8s} {'needed':>8s}")
    for i, n in enumerate(JOINT_NAMES):
        flag = "  <- outside real-robot q25-q75" if outside[i] else ""
        print(f"{n:24s} {tele[i]:8.3f} {q25[i]:7.3f} {q75[i]:7.3f} {model[i]:8.3f} {needed[i]:8.3f}{flag}")
    print(f"\nmax |needed| = {np.abs(needed).max():.3f} rad; "
          f"{int(outside.sum())}/{len(JOINT_NAMES)} joints outside the real-robot interquartile range")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(paths[0])), "right_hand_offset_from_live.json")
    with open(out, "w") as f:
        json.dump({
            "joint_names_right_hand_dims_36_to_57": JOINT_NAMES,
            "reference": {"dataset_dir": args.dataset_dir, "tele_episodes": list(args.tele_episodes),
                          "n_episodes": n_eps, "n_grasp_frames": n_frames, "top_frac": args.top_frac},
            "tele_grasp_median": tele.tolist(),
            "tele_grasp_q25": q25.tolist(),
            "tele_grasp_q75": q75.tolist(),
            "live": {"source": args.source, "grasp_window_s": args.grasp_window, "per_episode": per_episode},
            "model_grasp_median_live": model.tolist(),
            "offset_needed_tele_minus_model": needed.tolist(),
            "note": "ADD offset_needed_tele_minus_model to the absolute right-hand target the client "
                    "receives from the server (after the relative delta is added back); do not modify "
                    "the state fed to the model.",
        }, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
