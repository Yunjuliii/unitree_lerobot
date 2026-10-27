"""
Replay a 58-dim H2 + Sharpa joint column (default observation.h2_joints) from a LeRobot
dataset (v3.0 or v2.x) on the robot, reusing the hardware plumbing in replay_h2.py.

Column layout (checked against the column's names in meta/info.json):
    [0:14]  left arm 7 + right arm 7   (rad)
    [14:36] left Sharpa hand 22        (rad)
    [36:58] right Sharpa hand 22       (rad)

Per episode: arms go to frame 0 (the H2_ArmController rate limiter smooths the move), hands
ramp from their measured pose to frame 0, hold, play, hold. After the last episode (or Ctrl+C)
arms return home and hands ramp to neutral.

Usage (tv conda env, run from repo root; sharpa_dds_bridge must be running on Thor):
    # all episodes, half speed
    python unitree_lerobot/eval_robot/replay_h2_lerobot_column.py \
        --dataset-root /home/yunju/EgoCapInterface/cutton_bottle --speed 0.5

    # one at a time: wait for Enter before each episode
    python unitree_lerobot/eval_robot/replay_h2_lerobot_column.py \
        --dataset-root /home/yunju/EgoCapInterface/cutton_bottle --confirm

    # episode 1 only, 3 times, without gravity compensation (on by default)
    python unitree_lerobot/eval_robot/replay_h2_lerobot_column.py \
        --dataset-root /home/yunju/EgoCapInterface/cutton_bottle --episodes 1 --num-repeats 3 --no-gravity-comp
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_h2 import (  # noqa: E402
    ARM_SLICE, LEFT_EE_SLICE, RIGHT_EE_SLICE, SHARPA_DOF,
    connect_hand, disconnect_hand,
    _command_arm, _countdown, _make_arm_ik, _report_tracking, _run_episode, logger,
)
from replay_sharpa_hands import ramp, read_q_rad  # noqa: E402
from unitree_lerobot.utils.constants import H2_SHARPA_CONFIG  # noqa: E402

H2_DOF = 58


def load_episodes(root: Path, column: str, episodes: list[int] | None):
    """Return ({episode_index: (T, 58) float32}, fps, info)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.load(open(root / "meta" / "info.json"))
    feat = info["features"].get(column)
    if feat is None:
        raise ValueError(f"{column} not in {root}/meta/info.json features")
    names = feat.get("names")
    if feat["shape"] != [H2_DOF]:
        raise ValueError(f"{column} has shape {feat['shape']}, expected [{H2_DOF}]")
    # Joint names in the column must match the robot's 58-dim action order.
    if isinstance(names, list) and names != H2_SHARPA_CONFIG.motors:
        raise ValueError(f"{column} joint order differs from H2_SHARPA_CONFIG.motors")

    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    table = pa.concat_tables(
        [pq.read_table(f, columns=["episode_index", "frame_index", column]) for f in files]
    )
    ep_idx = np.asarray(table.column("episode_index"))
    frame_idx = np.asarray(table.column("frame_index"))
    q = np.stack(table.column(column).to_numpy(zero_copy_only=False)).astype(np.float32)

    out = {}
    for ep in (episodes if episodes else sorted(np.unique(ep_idx).tolist())):
        m = ep_idx == ep
        if not m.any():
            raise ValueError(f"episode {ep} not found in {root}")
        out[int(ep)] = q[m][np.argsort(frame_idx[m])]
    return out, float(info["fps"]), info


def go_to_first_frame(arm_ctrl, hands, first, hold_s, ramp_s, arm_ik):
    logger.info(f"Going to first frame (ramp {ramp_s:.1f}s, hold {hold_s}s)...")
    if arm_ctrl is not None:
        _command_arm(arm_ctrl, first[ARM_SLICE].astype(np.float64), arm_ik)
    if hands:
        target = {"left": first[LEFT_EE_SLICE], "right": first[RIGHT_EE_SLICE]}
        current = {s: read_q_rad(h) for s, h in hands.items()}
        ramp(hands, current, {s: target[s] for s in hands}, ramp_s, 30.0, "first frame")
    _countdown(hold_s, "holding first frame")


def main():
    ap = argparse.ArgumentParser(description="Replay a 58-dim H2 + Sharpa joint column from a LeRobot dataset.")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--column", default="observation.h2_joints")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all, in order")
    ap.add_argument("--num-repeats", type=int, default=1, help="repeats per episode")
    ap.add_argument("--start", type=int, default=0, help="first frame to play in each episode")
    ap.add_argument("--steps", type=int, default=None, help="play only this many frames per episode")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=None, help="default: dataset fps")
    ap.add_argument("--first-frame-hold-s", type=int, default=3)
    ap.add_argument("--hold-s", type=int, default=3, help="hold after each replay and at home")
    ap.add_argument("--ramp-s", type=float, default=2.0, help="hand ramp time to first frame / neutral")
    ap.add_argument("--confirm", action="store_true",
                    help="wait for Enter before each replay (s = skip this one, q = quit); the robot holds its pose")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arm", dest="arm", action="store_true", default=True)
    g.add_argument("--no-arm", dest="arm", action="store_false")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hands", dest="hands", action="store_true", default=True)
    g.add_argument("--no-hands", dest="hands", action="store_false")
    ap.add_argument("--hands-source", choices=["sdk", "dds"], default="dds")
    ap.add_argument("--hand-speed-coeff", type=float, default=0.5, help="SDK only")
    ap.add_argument("--hand-current-coeff", type=float, default=0.6, help="SDK only")

    # Defaults match run_replay.sh / the H2 eval launchers on this workstation.
    ap.add_argument("--dds-domain", type=int, default=0)
    ap.add_argument("--network-interface", default="eno2", metavar="IFACE")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--motion", dest="motion", action="store_true", default=True,
                   help="H2 motion mode (rt/arm_sdk), robot standing under WBC (default)")
    g.add_argument("--no-motion", dest="motion", action="store_false")
    ap.add_argument("--head-pitch-home", type=float, default=0.7, metavar="RAD")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gravity-comp", dest="gravity_comp", action="store_true", default=True,
                   help="tau_ff = G(q) (default on, as in the H2 eval launchers)")
    g.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false")
    ap.add_argument("--kp-low",   type=float, default=None)
    ap.add_argument("--kp-wrist", type=float, default=None)
    ap.add_argument("--kd-low",   type=float, default=None)
    ap.add_argument("--kd-wrist", type=float, default=None)
    ap.add_argument("--track-log-dir", default=".", help="where to save arm tracking logs")
    args = ap.parse_args()

    episodes, fps, info = load_episodes(args.dataset_root, args.column, args.episodes)
    end = None if args.steps is None else args.start + args.steps
    episodes = {ep: q[args.start:end] for ep, q in episodes.items()}
    fps = args.fps or fps
    dt = (1.0 / fps) / args.speed
    ik_meta = info.get("h2_arm_ik", {}).get(args.column.split(".")[-1], {})
    if "head_pitch_rad" in ik_meta:
        logger.info(f"Arm IK in this dataset assumed head pitch {ik_meta['head_pitch_rad']} rad "
                    f"(measured); commanding --head-pitch-home {args.head_pitch_home}.")
    for ep, q in episodes.items():
        step = np.degrees(np.abs(np.diff(q, axis=0)))
        logger.info(f"episode {ep}: {len(q)} frames ({len(q) * dt:.1f}s at {1 / dt:.1f} Hz), "
                    f"max step arm {step[:, ARM_SLICE].max():.1f} deg / hand {step[:, 14:].max():.1f} deg")

    if args.arm or (args.hands and args.hands_source == "dds"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(args.dds_domain, args.network_interface)

    # Gravity model before the arm controller starts publishing (see eval_h2_groot).
    arm_ik = _make_arm_ik(bool(args.arm and args.gravity_comp))
    arm_ctrl = None
    if args.arm:
        from teleop.robot_control.robot_arm import H2_ArmController
        logger.info(f"Initialising H2_ArmController (motion_mode={args.motion}, "
                    f"head_pitch_home={args.head_pitch_home})...")
        arm_ctrl = H2_ArmController(
            motion_mode=args.motion,
            head_pitch_home=args.head_pitch_home,
            kp_low=args.kp_low, kp_wrist=args.kp_wrist,
            kd_low=args.kd_low, kd_wrist=args.kd_wrist,
        )
        _command_arm(arm_ctrl, arm_ctrl.get_current_dual_arm_q(), arm_ik)
        arm_ctrl.speed_gradual_max()

    hands = {}
    if args.hands:
        for s in ("left", "right"):
            if args.hands_source == "dds":
                # No neutral command on connect (unlike replay_h2.connect_hand_dds): hands ramp from where they are.
                from teleop.robot_control.robot_hand_sharpa_dds import SharpaHandDDSClient
                hands[s] = SharpaHandDDSClient(s)
            else:
                hands[s] = connect_hand(s, args.hand_speed_coeff, args.hand_current_coeff)
                if hands[s] is None:
                    raise RuntimeError(f"Failed to connect {s} hand")
        time.sleep(1.0)  # first state messages
    left_hand, right_hand = hands.get("left"), hands.get("right")

    try:
        quit_all = False
        for ep, q in episodes.items():
            actions = torch.from_numpy(q)
            for rep in range(args.num_repeats):
                if args.confirm:
                    ans = input(f"\n>>> Episode {ep} replay {rep + 1}/{args.num_repeats}: "
                                "Enter = play, s = skip, q = quit: ").strip().lower()
                    if ans == "q":
                        quit_all = True
                        break
                    if ans == "s":
                        continue
                logger.info(f"\n=== Episode {ep}  replay {rep + 1}/{args.num_repeats} ===")
                go_to_first_frame(arm_ctrl, hands, q[0], args.first_frame_hold_s, args.ramp_s, arm_ik)
                q_target, q_meas = _run_episode(arm_ctrl, left_hand, right_hand, actions, dt, arm_ik=arm_ik)
                if arm_ctrl is not None:
                    tag = "gc_on" if args.gravity_comp else "gc_off"
                    save_path = os.path.join(
                        os.path.expanduser(args.track_log_dir),
                        f"replay_track_{args.dataset_root.name}_ep{ep}_{tag}_rep{rep + 1}_"
                        f"{time.strftime('%Y%m%d_%H%M%S')}.npz",
                    )
                    _report_tracking(q_target, q_meas, args.gravity_comp, save_path)
                _countdown(args.hold_s, "holding last frame")
            if quit_all:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if hands:
            current = {s: read_q_rad(h) for s, h in hands.items()}
            ramp(hands, current, {s: np.zeros(SHARPA_DOF) for s in hands}, args.ramp_s, 30.0, "neutral")
        if arm_ctrl is not None:
            logger.info("Arms going home...")
            _command_arm(arm_ctrl, np.zeros(14), arm_ik)
            arm_ctrl.ctrl_dual_arm_go_home()
            _countdown(args.hold_s, "holding home")
        if args.hands_source == "sdk":
            for s, h in hands.items():
                disconnect_hand(h, s)
    logger.info("Replay finished.")


if __name__ == "__main__":
    main()
