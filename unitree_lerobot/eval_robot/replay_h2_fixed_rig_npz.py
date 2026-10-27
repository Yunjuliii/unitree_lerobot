"""
Replay an H2 + Sharpa "fixed rig" trajectory .npz (not a LeRobot dataset) on the
robot, by reusing the hardware plumbing in replay_h2.py.

The .npz (e.g. h2_sharpa_fixed_rig_trajectory.npz) stores a (T, 75) joint_positions
array in H2 action order:
    0:12   legs            (ignored here)
    12:15  waist           (ignored here)
    15:17  head pitch/yaw  (ignored here — use --head-pitch-home)
    17:31  left+right arm  (14) -> ARM_SLICE
    31:75  44 Sharpa finger joints, interleaved left/right by name

This script only drives arm + hands, mirroring replay_h2.py's --no-legs behavior
(replay_h2.py never controls legs/waist to begin with).

Usage (tv conda env, run from repo root):
    python unitree_lerobot/eval_robot/replay_h2_fixed_rig_npz.py \
        --npz-path /home/yunju/xr_teleoperate/h2_sharpa_fixed_rig_trajectory.npz \
        --hands-source dds --motion --head-pitch-home 0.837
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_h2 import (  # noqa: E402
    ARM_SLICE, SHARPA_DOF,
    connect_hand, disconnect_hand, connect_hand_dds, disconnect_hand_dds,
    _go_home, _go_to_first_frame, _run_episode, _report_tracking,
    _make_arm_ik, _command_arm, _dbg, logger,
)

# ── column mapping: joint_positions (T, 75) -> action (T, 58) ────────────────
# Target order (58-dim action, per info.json features.joints.names / H2_with_sharpa_hands.urdf):
#   0:14  arm  = left(shoulder_pitch,shoulder_roll,shoulder_yaw,elbow,wrist_roll,wrist_pitch,wrist_yaw)
#                + right(same 7, mirrored)
#   14:36 left hand  = thumb(CMC_FE,CMC_AA,MCP_FE,MCP_AA,IP) + index/middle/ring(MCP_FE,MCP_AA,PIP,DIP)
#                       + pinky(CMC,MCP_FE,MCP_AA,PIP,DIP)   [5+4+4+4+5 = 22]
#   36:58 right hand = same order as left, "right_" prefix
# joint_positions' own column order (npz joint_names) groups by joint TYPE across fingers
# (all *_MCP_FE together, etc.), not by finger — so indices are resolved by name, not
# assumed to be contiguous/ordered the same way.
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINT_SUFFIXES = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
]
assert len(ARM_JOINT_NAMES) == 14
assert len(HAND_JOINT_SUFFIXES) == SHARPA_DOF


def load_npz_actions(npz_path: str, episode_idx: int | None) -> tuple[torch.Tensor, dict]:
    """Return (actions (T, 58), meta) where meta holds fps / head_pitch_rad recorded in the npz."""
    _dbg(f"load_npz_actions: loading {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    meta = {
        "fps": float(data["fps"]) if "fps" in data.files else None,
        "head_pitch_rad": float(data["head_pitch_rad"]) if "head_pitch_rad" in data.files else None,
    }
    joint_positions = data["joint_positions"]  # (T, 75)
    joint_names = [str(n) for n in data["joint_names"]]
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    arm_idx = [name_to_idx[n] for n in ARM_JOINT_NAMES]
    left_hand_idx = [name_to_idx[f"left_{s}"] for s in HAND_JOINT_SUFFIXES]
    right_hand_idx = [name_to_idx[f"right_{s}"] for s in HAND_JOINT_SUFFIXES]

    if episode_idx is not None and "episode_index" in data.files:
        mask = data["episode_index"] == episode_idx
        joint_positions = joint_positions[mask]
        _dbg(f"load_npz_actions: filtered to episode {episode_idx}, {len(joint_positions)} frames")

    T = joint_positions.shape[0]
    actions = np.zeros((T, 58), dtype=np.float32)
    actions[:, 0:14] = joint_positions[:, arm_idx]
    actions[:, 14:36] = joint_positions[:, left_hand_idx]
    actions[:, 36:58] = joint_positions[:, right_hand_idx]

    if "joint_limit_violation" in data.files:
        n_bad = int(np.sum(data["joint_limit_violation"]))
        if n_bad:
            logger.warning(f"{n_bad}/{T} frames have joint_limit_violation=True")

    logger.info(f"Loaded {T} frames from {npz_path}, action shape={actions.shape}, meta={meta}")
    return torch.from_numpy(actions), meta


def replay(args):
    _dbg("replay() start")
    actions, meta = load_npz_actions(args.npz_path, args.episode)
    if args.steps is not None and args.steps > 0:
        actions = actions[:args.steps]
        logger.info(f"Truncated to first {len(actions)} frames (--steps={args.steps}).")
    fps = args.fps if args.fps is not None else (meta["fps"] or 30.0)
    if args.head_pitch_home is None:
        args.head_pitch_home = meta["head_pitch_rad"] if meta["head_pitch_rad"] is not None else 0.837
    dt = (1.0 / fps) / args.speed
    _dbg(f"dt = {dt:.4f}s  (speed={args.speed}, fps={fps})")
    if args.urdf_path:
        os.environ["H2_IK_URDF"] = os.path.expanduser(args.urdf_path)

    need_dds = args.arm or (args.hands and args.hands_source == "dds")
    if need_dds:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(args.dds_domain, networkInterface=args.network_interface)
        logger.info(
            f"DDS ChannelFactory initialised on domain {args.dds_domain} "
            f"(network_interface={args.network_interface})."
        )

    arm_ctrl = None
    if args.arm:
        from teleop.robot_control.robot_arm import H2_ArmController
        logger.info(
            f"Initialising H2_ArmController (motion_mode={args.motion}, "
            f"head_pitch_home={args.head_pitch_home})..."
        )
        arm_ctrl = H2_ArmController(
            motion_mode=args.motion,
            head_pitch_home=args.head_pitch_home,
            kp_low=args.kp_low, kp_wrist=args.kp_wrist,
            kd_low=args.kd_low, kd_wrist=args.kd_wrist,
        )
        arm_ctrl.speed_gradual_max()
        logger.info("H2_ArmController ready.")

    arm_ik = _make_arm_ik(bool(args.arm and args.gravity_comp))
    if args.arm and not args.gravity_comp:
        logger.info("Gravity compensation disabled (tau_ff = 0). Pass --gravity-comp to enable.")

    left_hand = None
    right_hand = None
    if args.hands:
        if args.hands_source == "dds":
            left_hand = connect_hand_dds("left")
            right_hand = connect_hand_dds("right")
        else:
            left_hand = connect_hand("left", args.hand_speed_coeff, args.hand_current_coeff)
            right_hand = connect_hand("right", args.hand_speed_coeff, args.hand_current_coeff)
        time.sleep(2.0)

    logger.info(f"Hardware ready.  arm={args.arm}  hands={args.hands}  repeats={args.num_repeats}")

    _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s)

    first_action = actions[0].numpy()

    interrupted = False
    try:
        for rep in range(args.num_repeats):
            logger.info(f"\n=== Replay {rep + 1}/{args.num_repeats} ===")
            _go_to_first_frame(arm_ctrl, left_hand, right_hand,
                               first_action, hold_s=args.first_frame_hold_s, arm_ik=arm_ik)
            q_target_log, q_meas_log = _run_episode(arm_ctrl, left_hand, right_hand, actions, dt, arm_ik=arm_ik)
            if arm_ctrl is not None:
                tag = "gc_on" if args.gravity_comp else "gc_off"
                stem = os.path.splitext(os.path.basename(args.npz_path))[0]
                save_path = os.path.join(
                    os.path.expanduser(args.track_log_dir),
                    f"replay_track_{stem}_{tag}_rep{rep + 1}_{time.strftime('%Y%m%d_%H%M%S')}.npz",
                )
                _report_tracking(q_target_log, q_meas_log, args.gravity_comp, save_path)

            if rep < args.num_repeats - 1:
                _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s)

    except KeyboardInterrupt:
        interrupted = True
        logger.info("Interrupted by user.")

    if not interrupted and args.hold_s > 0:
        logger.info(f"All repeats done. Holding last action for {args.hold_s}s...")
        last_arm_q = actions[-1].numpy()[ARM_SLICE].astype(np.float64)
        for i in range(args.hold_s, 0, -1):
            print(f"  holding: {i}s ", end="\r", flush=True)
            if arm_ctrl is not None:
                _command_arm(arm_ctrl, last_arm_q, arm_ik)
            time.sleep(1.0)
        print("  done.          ")

    if args.hands:
        if args.hands_source == "dds":
            disconnect_hand_dds(left_hand, "left")
            disconnect_hand_dds(right_hand, "right")
        else:
            disconnect_hand(left_hand, "left")
            disconnect_hand(right_hand, "right")
    logger.info("Replay finished.")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Replay an H2 + Sharpa fixed-rig trajectory .npz (arm + hands only)."
    )
    ap.add_argument("--npz-path", required=True, help="Path to the fixed-rig trajectory .npz")
    ap.add_argument("--episode", type=int, default=None,
                     help="Filter to a single episode_index in the npz (default: use all frames).")
    ap.add_argument("--num-repeats", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None,
                     help="Playback fps. Default: the fps stored in the npz (fallback 30).")
    ap.add_argument("--track-log-dir", default=".",
                     help="Directory for the per-replay tracking log .npz (target vs measured arm q). Default: cwd.")
    ap.add_argument("--first-frame-hold-s", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--hold-s", type=int, default=20)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arm", dest="arm", action="store_true", default=True)
    g.add_argument("--no-arm", dest="arm", action="store_false")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hands", dest="hands", action="store_true", default=True)
    g.add_argument("--no-hands", dest="hands", action="store_false")

    ap.add_argument("--hands-source", choices=["sdk", "dds"], default="sdk")
    ap.add_argument("--dds-domain", type=int, default=0)
    ap.add_argument("--network-interface", type=str, default=None,
                     help="Network interface for DDS communication, e.g. eno2. "
                          "If None, uses the SDK default interface.")

    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--head-pitch-home", type=float, default=None, metavar="RAD",
                     help="Head pitch held during replay. Default: head_pitch_rad stored in the npz (fallback 0.837).")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gravity-comp", dest="gravity_comp", action="store_true", default=False,
                    help="Enable gravity compensation: tau_ff = G(q) via H2_ArmIK.compute_tauff "
                         "(same code path as teleop / eval_h2_groot / replay_h2). Default: off.")
    g.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false")
    ap.add_argument("--urdf-path", type=str, default=None,
                     help="Override the URDF used for gravity compensation (sets H2_IK_URDF). "
                          "Default: xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf.")

    ap.add_argument("--kp-low", type=float, default=None)
    ap.add_argument("--kp-wrist", type=float, default=None)
    ap.add_argument("--kd-low", type=float, default=None)
    ap.add_argument("--kd-wrist", type=float, default=None)

    ap.add_argument("--hand-speed-coeff", type=float, default=0.5)
    ap.add_argument("--hand-current-coeff", type=float, default=0.6)
    return ap.parse_args()


if __name__ == "__main__":
    replay(parse_args())
