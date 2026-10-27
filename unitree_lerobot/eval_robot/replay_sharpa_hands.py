"""
Replay recorded Sharpa hand joint angles only (no arm, no head, no legs).

.parquet, or .csv with a header row, with per-joint columns left_<joint> / right_<joint>
(e.g. EgoCapInterface *_sharpa_joints_rad.{parquet,csv}) is read by column name, both hands.

Other accepted data files (.npy / .npz / .csv / .txt), one row per frame:
    (T, 22)  one hand          -> which hand is chosen by --side
    (T, 44)  [left 22 | right 22]
    (T, 58)  H2 action layout  -> left 14:36, right 36:58 (arm columns ignored)
For .npz, pass --key to pick the array; otherwise keys "left"/"right" (each (T, 22))
or the only array in the file are used.

Safety: before playback each hand ramps from its current measured pose to frame 0
over --ramp-s seconds; after playback it ramps back to neutral (all zeros).

Usage (tv conda env, run from repo root; sharpa_dds_bridge must be running on Thor):
    python unitree_lerobot/eval_robot/replay_sharpa_hands.py \
        --data hand_traj.npz --network-interface eno2

    # one-hand file in degrees, half speed
    python unitree_lerobot/eval_robot/replay_sharpa_hands.py \
        --data right_hand.csv --side right --unit deg --speed 0.5 --network-interface eno2
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_h2 import (  # noqa: E402
    LEFT_EE_SLICE, RIGHT_EE_SLICE, SHARPA_DOF,
    connect_hand, disconnect_hand, logger,
)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_array(path: str, key: str | None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        data = np.load(path)
        if key is not None:
            return data[key]
        if "left" in data.files and "right" in data.files:
            return {"left": data["left"], "right": data["right"]}
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError(f"{path} has several arrays {data.files}; pick one with --key")
    if ext == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        names = table.column_names
        if "frame_index" in names:
            table = table.take(np.argsort(np.asarray(table.column("frame_index"))))
        if key is not None:
            return np.stack(table.column(key).to_numpy(zero_copy_only=False))
        # Per-joint columns named left_<joint> / right_<joint>, in file (joint) order.
        out = {}
        for s in ("left", "right"):
            cols = [c for c in names if c.startswith(f"{s}_")]
            if cols:
                out[s] = np.stack([np.asarray(table.column(c), dtype=np.float64) for c in cols], axis=1)
        if out:
            return out
        if "action" in names:  # LeRobot episode parquet
            return np.stack(table.column("action").to_numpy(zero_copy_only=False))
        raise ValueError(f"{path}: no left_*/right_* columns or 'action' column; pick one with --key")
    if ext in (".csv", ".txt"):
        delim = "," if ext == ".csv" else None
        with open(path) as f:
            header = [c.strip() for c in f.readline().split(delim)]
        try:
            float(header[0])
            return np.loadtxt(path, delimiter=delim)
        except ValueError:
            pass  # header row
        data = np.atleast_2d(np.loadtxt(path, delimiter=delim, skiprows=1))
        if "frame_index" in header:
            data = data[np.argsort(data[:, header.index("frame_index")])]
        if key is not None:
            return data[:, [header.index(key)]]
        # Same per-joint column naming as the .parquet case.
        out = {}
        for s in ("left", "right"):
            idx = [i for i, c in enumerate(header) if c.startswith(f"{s}_")]
            if idx:
                out[s] = data[:, idx]
        return out if out else data
    raise ValueError(f"Unsupported file type: {ext}")


def load_hand_traj(path: str, key: str | None, side: str, unit: str) -> dict:
    """Return {"left": (T, 22) rad, "right": (T, 22) rad} (only the sides present)."""
    raw = _load_array(os.path.expanduser(path), key)
    if isinstance(raw, dict):
        traj = raw
    else:
        raw = np.atleast_2d(raw)
        ncol = raw.shape[1]
        if ncol == SHARPA_DOF:
            if side == "both":
                raise ValueError("Data has 22 columns (one hand); pass --side left or --side right")
            traj = {side: raw}
        elif ncol == 2 * SHARPA_DOF:
            traj = {"left": raw[:, :SHARPA_DOF], "right": raw[:, SHARPA_DOF:]}
        elif ncol == 58:
            traj = {"left": raw[:, LEFT_EE_SLICE], "right": raw[:, RIGHT_EE_SLICE]}
        else:
            raise ValueError(f"Expected 22, 44 or 58 columns, got shape {raw.shape}")

    if side != "both":
        if side not in traj:
            raise ValueError(f"--side {side} requested but data only has {list(traj)}")
        traj = {side: traj[side]}

    out = {}
    for s, q in traj.items():
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != SHARPA_DOF:
            raise ValueError(f"{s} hand data must be (T, {SHARPA_DOF}), got {q.shape}")
        out[s] = np.radians(q) if unit == "deg" else q
    lengths = {s: len(q) for s, q in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Left/right frame counts differ: {lengths}")
    return out


# ── hand connection ───────────────────────────────────────────────────────────

def connect(side: str, source: str, speed_coeff: float, current_coeff: float):
    if source == "sdk":
        return connect_hand(side, speed_coeff, current_coeff)
    # Build the DDS client directly (no neutral command on connect, unlike
    # replay_h2.connect_hand_dds) so the hand ramps smoothly from where it is.
    from teleop.robot_control.robot_hand_sharpa_dds import SharpaHandDDSClient
    return SharpaHandDDSClient(side)


def read_q_rad(hand) -> np.ndarray:
    err, deg = hand.get_joint_position_degree()
    if getattr(err, "code", err) != 0:
        raise RuntimeError(f"get_joint_position_degree failed: {err}")
    return np.array([math.radians(d) for d in deg[:SHARPA_DOF]])


def send(hands: dict, q_by_side: dict):
    for s, hand in hands.items():
        err = hand.set_joint_position(q_by_side[s].tolist(), False)
        # DDS client returns None; SDK returns an Error object.
        if err is not None and getattr(err, "code", 0) != 0:
            logger.warning(f"[{s}] set_joint_position error: {err.message}")


def ramp(hands: dict, q_from: dict, q_to: dict, duration_s: float, rate_hz: float, label: str):
    n = max(1, int(duration_s * rate_hz))
    logger.info(f"Ramping to {label} over {duration_s:.1f}s ...")
    for i in range(1, n + 1):
        a = i / n
        send(hands, {s: (1 - a) * q_from[s] + a * q_to[s] for s in hands})
        time.sleep(1.0 / rate_hz)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Replay recorded Sharpa hand joint angles (hands only).")
    ap.add_argument("--data", required=True, help=".parquet / .npy / .npz / .csv / .txt joint-angle file")
    ap.add_argument("--key", default=None, help="array name inside an .npz / column name inside a .parquet")
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    ap.add_argument("--unit", choices=["rad", "deg"], default="rad", help="unit of the data (default: rad)")
    ap.add_argument("--fps", type=float, default=30.0, help="recording rate (default: 30)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--start", type=int, default=0, help="first frame to play")
    ap.add_argument("--steps", type=int, default=None, help="play only this many frames")
    ap.add_argument("--num-repeats", type=int, default=1)
    ap.add_argument("--ramp-s", type=float, default=2.0, help="ramp time to first frame / neutral")
    ap.add_argument("--hold-s", type=float, default=1.0, help="hold time at first and last frame")
    ap.add_argument("--hands-source", choices=["sdk", "dds"], default="dds",
                    help="'dds' = via sharpa_dds_bridge on Thor (default); 'sdk' = direct SDK on Thor")
    ap.add_argument("--dds-domain", type=int, default=0)
    ap.add_argument("--network-interface", default=None, metavar="IFACE", help="e.g. eno2")
    ap.add_argument("--hand-speed-coeff", type=float, default=0.5, help="SDK only")
    ap.add_argument("--hand-current-coeff", type=float, default=0.6, help="SDK only")
    args = ap.parse_args()

    traj = load_hand_traj(args.data, args.key, args.side, args.unit)
    end = None if args.steps is None else args.start + args.steps
    traj = {s: q[args.start:end] for s, q in traj.items()}
    T = len(next(iter(traj.values())))
    if T == 0:
        raise ValueError("No frames to play after --start/--steps")
    dt = 1.0 / (args.fps * args.speed)
    for s, q in traj.items():
        logger.info(f"[{s}] {T} frames, range [{q.min():.3f}, {q.max():.3f}] rad")
    logger.info(f"Playback: {T * dt:.1f}s at {1 / dt:.1f} Hz")

    if args.hands_source == "dds":
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(args.dds_domain, args.network_interface)

    hands = {}
    for s in traj:
        hand = connect(s, args.hands_source, args.hand_speed_coeff, args.hand_current_coeff)
        if hand is None:
            raise RuntimeError(f"Failed to connect {s} hand")
        hands[s] = hand
    time.sleep(1.0)  # let the first state messages arrive

    zeros = {s: np.zeros(SHARPA_DOF) for s in hands}
    first = {s: traj[s][0] for s in hands}
    last = first
    try:
        for rep in range(args.num_repeats):
            logger.info(f"=== Replay {rep + 1}/{args.num_repeats} ===")
            current = {s: read_q_rad(h) for s, h in hands.items()}
            ramp(hands, current, first, args.ramp_s, args.fps, "first frame")
            time.sleep(args.hold_s)

            t_next = time.perf_counter()
            for i in range(T):
                last = {s: traj[s][i] for s in hands}
                send(hands, last)
                if (i + 1) % 30 == 0:
                    print(f"  frame {i + 1}/{T}", end="\r", flush=True)
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
            print(f"  frame {T}/{T}")
            time.sleep(args.hold_s)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        ramp(hands, last, zeros, args.ramp_s, args.fps, "neutral")
        if args.hands_source == "sdk":
            for s, hand in hands.items():
                disconnect_hand(hand, s)
    logger.info("Done.")


if __name__ == "__main__":
    main()
