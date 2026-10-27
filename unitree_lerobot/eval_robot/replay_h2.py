"""
Replay a LeRobot episode on the H2 robot with Sharpa dexterous hands.

Usage (tv conda env, run from repo root):
    # replay arm + hands via direct SDK (must run on Thor, no bridge active)
    python unitree_lerobot/eval_robot/replay_h2.py \
        --dataset-dir /home/nvidia/lerobot --num-repeats 3

    # replay through the DDS bridge on Thor (run from workstation):
    python unitree_lerobot/eval_robot/replay_h2.py \
        --dataset-dir /home/nvidia/lerobot --hands-source dds

    # full replay with the robot standing on its legs (WBC yields arms via rt/arm_sdk):
    python unitree_lerobot/eval_robot/replay_h2.py \
        --dataset-dir /home/nvidia/lerobot --hands-source dds --motion

    # arm only  (no Sharpa hardware required)
    python unitree_lerobot/eval_robot/replay_h2.py \
        --dataset-dir /home/nvidia/lerobot --no-hands

    # hands only
    python unitree_lerobot/eval_robot/replay_h2.py \
        --dataset-dir /home/nvidia/lerobot --no-arm

    # specific episode, half speed
    python unitree_lerobot/eval_robot/replay_h2.py \
        --dataset-dir /home/nvidia/lerobot --episode 0 --speed 0.5
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_XR_ROOT   = os.path.expanduser("~/xr_teleoperate")
_SHARPA_SDK = os.path.expanduser("~/Sharpa/SharpaWaveSDK_4.6.6/python")

for p in [_REPO_ROOT, _XR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── action layout (H2 + Sharpa, 58-dim) ──────────────────────────────────────
ARM_SLICE       = slice(0, 14)
LEFT_EE_SLICE   = slice(14, 36)
RIGHT_EE_SLICE  = slice(36, 58)
SHARPA_DOF      = 22

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger = logging_mp.getLogger(__name__)


def _dbg(msg: str):
    """Unbuffered timestamped debug print — survives logging buffering."""
    print(f"[DBG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Sharpa hand helpers ───────────────────────────────────────────────────────

def _sharpa_path():
    if _SHARPA_SDK not in sys.path:
        sys.path.insert(0, _SHARPA_SDK)


def connect_hand(side: str, speed_coeff: float = 0.5, current_coeff: float = 0.6):
    _dbg(f"connect_hand[{side}]: importing sharpa ...")
    _sharpa_path()
    from sharpa import SharpaWaveManager, ControlMode, ControlSource, HandSide
    _dbg(f"connect_hand[{side}]: sharpa imported")

    hand_side = HandSide.LEFT if side == "left" else HandSide.RIGHT
    _dbg(f"connect_hand[{side}]: getting manager instance ...")
    manager   = SharpaWaveManager.get_instance()
    _dbg(f"connect_hand[{side}]: got manager, sleeping 1.0s ...")
    time.sleep(1.0)

    logger.info(f"[{side}] connecting to {hand_side.name} hand...")
    _dbg(f"connect_hand[{side}]: manager.connect({hand_side.name}) ...")
    try:
        hand = manager.connect(hand_side)
        _dbg(f"connect_hand[{side}]: manager.connect returned (hand={hand is not None})")
    except Exception as e:
        logger.warning(f"[{side}] connect failed: {e}  — skipping")
        return None

    if hand is None:
        logger.warning(f"[{side}] manager.connect returned None — skipping")
        return None

    for fn, val in [
        ("set_control_mode",   ControlMode.POSITION),
        ("set_control_source", ControlSource.SDK),
    ]:
        err = getattr(hand, fn)(val)
        if err.code != 0:
            logger.error(f"[{side}] {fn} failed: {err.message}")
            return None

    _dbg(f"connect_hand[{side}]: set_speed_coeff({speed_coeff}) ...")
    hand.set_speed_coeff(speed_coeff)
    _dbg(f"connect_hand[{side}]: set_current_coeff({current_coeff}) ...")
    hand.set_current_coeff(current_coeff)
    _dbg(f"connect_hand[{side}]: hand.start() ...")
    hand.start()
    _dbg(f"connect_hand[{side}]: sending neutral set_joint_position ...")
    hand.set_joint_position([0.0] * SHARPA_DOF, True)
    _dbg(f"connect_hand[{side}]: neutral set; ready")
    logger.info(f"[{side}] Sharpa hand ready.")
    return hand


def disconnect_hand(hand, side: str):
    if hand is None:
        return
    _sharpa_path()
    from sharpa import SharpaWaveManager
    try:
        hand.set_joint_position([0.0] * SHARPA_DOF, True)
        time.sleep(0.5)
        hand.stop()
    except Exception as e:
        logger.warning(f"[{side}] stop error: {e}")
    try:
        SharpaWaveManager.get_instance().disconnect_all()
    except Exception:
        pass
    logger.info(f"[{side}] disconnected.")


# ── Sharpa hand helpers — DDS bridge path ────────────────────────────────────

def connect_hand_dds(side: str):
    """Construct a SharpaHandDDSClient that talks to the bridge on Thor."""
    _dbg(f"connect_hand_dds[{side}]: importing SharpaHandDDSClient ...")
    from teleop.robot_control.robot_hand_sharpa_dds import SharpaHandDDSClient
    _dbg(f"connect_hand_dds[{side}]: import done")
    logger.info(f"[{side}] creating DDS client (bridge on Thor must be running)...")
    _dbg(f"connect_hand_dds[{side}]: constructing SharpaHandDDSClient ...")
    client = SharpaHandDDSClient(side)
    _dbg(f"connect_hand_dds[{side}]: constructed; sending neutral target ...")
    # Send a neutral target so the bridge applies it immediately.
    client.set_joint_position([0.0] * SHARPA_DOF, False)
    _dbg(f"connect_hand_dds[{side}]: neutral sent; ready")
    logger.info(f"[{side}] DDS client ready.")
    return client


def disconnect_hand_dds(client, side: str):
    if client is None:
        return
    try:
        client.go_neutral()
        time.sleep(0.5)
    except Exception as e:
        logger.warning(f"[{side}] DDS go_neutral error: {e}")
    logger.info(f"[{side}] DDS client released.")


# ── dataset loading ───────────────────────────────────────────────────────────

def load_episode(dataset_dir: str, episode_idx: int) -> tuple[torch.Tensor, float]:
    """Return (actions (T, 58), fps) for the requested episode, read straight from the LeRobot v2.x parquet."""
    import json
    import pyarrow.parquet as pq

    dataset_dir = os.path.expanduser(dataset_dir)
    with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
        info = json.load(f)
    fps = float(info["fps"])
    chunk = episode_idx // int(info.get("chunks_size", 1000))
    parquet_path = os.path.join(
        dataset_dir, info["data_path"].format(episode_chunk=chunk, episode_index=episode_idx)
    )
    _dbg(f"load_episode: reading {parquet_path}")
    if not os.path.isfile(parquet_path):
        raise ValueError(f"Episode {episode_idx} not found: {parquet_path}")

    table = pq.read_table(parquet_path, columns=["action", "frame_index"])
    order = np.argsort(np.asarray(table.column("frame_index").to_pylist()))
    actions_np = np.stack(table.column("action").to_numpy(zero_copy_only=False))[order].astype(np.float32)
    actions = torch.from_numpy(actions_np)
    logger.info(f"Loaded episode {episode_idx}: {len(actions)} frames, shape={tuple(actions.shape)}, fps={fps}")
    return actions, fps


# ── control helpers ───────────────────────────────────────────────────────────

def _make_arm_ik(enabled: bool):
    if not enabled:
        return None
    from teleop.robot_control.robot_arm_ik import H2_ArmIK
    logger.info("Loading H2_ArmIK for Pinocchio RNEA gravity compensation...")
    arm_ik = H2_ArmIK(Visualization=False)
    logger.info("Gravity compensation enabled (tau_ff = G(q)).")
    return arm_ik


def _command_arm(arm_ctrl, q_target: np.ndarray, arm_ik):
    q = np.asarray(q_target, dtype=np.float64).reshape(-1)
    tau = arm_ik.compute_tauff(q) if arm_ik is not None else np.zeros(14, dtype=np.float64)
    arm_ctrl.ctrl_dual_arm(q, tau)


def _countdown(seconds: int, label: str):
    for i in range(seconds, 0, -1):
        print(f"  {label}: {i}s ", end="\r", flush=True)
        time.sleep(1.0)
    print(f"  {label}: done.          ")


def _go_home(arm_ctrl, left_hand, right_hand, hold_s: int):
    logger.info(f"Going to home position (hold {hold_s}s)...")
    _dbg(f"_go_home: arm_ctrl={arm_ctrl is not None} left={left_hand is not None} right={right_hand is not None} hold_s={hold_s}")
    if arm_ctrl is not None:
        _dbg("_go_home: calling arm_ctrl.ctrl_dual_arm_go_home() ...")
        arm_ctrl.ctrl_dual_arm_go_home()
        _dbg("_go_home: ctrl_dual_arm_go_home() returned")
    for side, hand in [("left", left_hand), ("right", right_hand)]:
        if hand is not None:
            _dbg(f"_go_home: sending neutral to {side} hand ...")
            hand.set_joint_position([0.0] * SHARPA_DOF, True)
            _dbg(f"_go_home: {side} neutral done")
    _countdown(hold_s, "holding home")


def _go_to_first_frame(arm_ctrl, left_hand, right_hand, first_action: np.ndarray, hold_s: int, arm_ik=None):
    """Smoothly move arm + hands to the first frame of the episode and hold.

    Mirrors eval_h2_groot._go_to_init_state. With WBC active (motion_mode), this
    avoids the discontinuity that would otherwise occur when playback jumps from
    home-zeros to action[0]. The H2_ArmController's 250Hz thread + clip_arm_q_target
    rate-limiter handles the actual interpolation.
    """
    logger.info(f"Going to first-frame posture (hold {hold_s}s)...")
    _dbg("_go_to_first_frame: start")
    if arm_ctrl is not None:
        arm_q_target = first_action[ARM_SLICE].astype(np.float64)
        _dbg(f"_go_to_first_frame: arm target = {arm_q_target.tolist()}")
        _command_arm(arm_ctrl, arm_q_target, arm_ik)
    if left_hand is not None:
        lh = first_action[LEFT_EE_SLICE].tolist()
        _dbg("_go_to_first_frame: left hand target (blocking) ...")
        left_hand.set_joint_position(lh, True)
        _dbg("_go_to_first_frame: left hand reached target")
    if right_hand is not None:
        rh = first_action[RIGHT_EE_SLICE].tolist()
        _dbg("_go_to_first_frame: right hand target (blocking) ...")
        right_hand.set_joint_position(rh, True)
        _dbg("_go_to_first_frame: right hand reached target")
    _countdown(hold_s, "holding first frame")


def _run_episode(arm_ctrl, left_hand, right_hand, actions: torch.Tensor, dt: float, arm_ik=None):
    """Play the episode; return (q_target, q_measured) arrays of shape (T, 14) for tracking analysis."""
    T = len(actions)
    _dbg(f"_run_episode: T={T}  dt={dt:.4f}s")
    q_target_log = np.zeros((T, 14))
    q_meas_log   = np.zeros((T, 14))
    for frame_idx in range(T):
        _dbg(f"frame {frame_idx}: start")
        t0     = time.time()
        action = actions[frame_idx].numpy()

        if arm_ctrl is not None:
            _dbg(f"frame {frame_idx}: ctrl_dual_arm ...")
            q_target = action[ARM_SLICE].astype(np.float64)
            q_meas_log[frame_idx]   = arm_ctrl.get_current_dual_arm_q()
            q_target_log[frame_idx] = q_target
            _command_arm(arm_ctrl, q_target, arm_ik)
            _dbg(f"frame {frame_idx}: ctrl_dual_arm done")

        if left_hand is not None:
            _dbg(f"frame {frame_idx}: left hand set_joint_position ...")
            err = left_hand.set_joint_position(action[LEFT_EE_SLICE].tolist(), False)
            _dbg(f"frame {frame_idx}: left hand done")
            # DDS client returns None; SDK returns an Error object.
            if err is not None and getattr(err, "code", 0) != 0:
                logger.warning(f"[frame {frame_idx}] left hand error: {err.message}")

        if right_hand is not None:
            _dbg(f"frame {frame_idx}: right hand set_joint_position ...")
            err = right_hand.set_joint_position(action[RIGHT_EE_SLICE].tolist(), False)
            _dbg(f"frame {frame_idx}: right hand done")
            if err is not None and getattr(err, "code", 0) != 0:
                logger.warning(f"[frame {frame_idx}] right hand error: {err.message}")

        elapsed = time.time() - t0
        sleep_s = max(0.0, dt - elapsed)
        _dbg(f"frame {frame_idx}: work={elapsed*1000:.1f}ms sleep={sleep_s*1000:.1f}ms")
        time.sleep(sleep_s)

        if (frame_idx + 1) % 30 == 0:
            logger.info(f"  frame {frame_idx + 1}/{T}  loop_dt={1000*(elapsed + sleep_s):.1f}ms")
    _dbg(f"_run_episode: completed all {T} frames")
    return q_target_log, q_meas_log


_ARM_JOINT_LABELS = [
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow", "L_wr_roll", "L_wr_pitch", "L_wr_yaw",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow", "R_wr_roll", "R_wr_pitch", "R_wr_yaw",
]


def _report_tracking(q_target: np.ndarray, q_meas: np.ndarray, gravity_comp: bool, save_path: str):
    """Print per-joint tracking error (target vs measured) and save the raw logs."""
    err = q_meas - q_target
    mean_abs = np.abs(err).mean(axis=0)
    max_abs  = np.abs(err).max(axis=0)
    bias     = err.mean(axis=0)
    tag = "gravity_comp=ON " if gravity_comp else "gravity_comp=OFF"
    logger.info(f"\n=== Arm tracking error [{tag}]  (measured - target, rad) ===")
    logger.info(f"  {'joint':<11}{'mean|e|':>9}{'max|e|':>9}{'bias':>9}")
    for i, name in enumerate(_ARM_JOINT_LABELS):
        logger.info(f"  {name:<11}{mean_abs[i]:9.4f}{max_abs[i]:9.4f}{bias[i]:+9.4f}")
    logger.info(f"  {'ALL':<11}{mean_abs.mean():9.4f}{max_abs.max():9.4f}")
    np.savez(save_path, q_target=q_target, q_meas=q_meas, gravity_comp=gravity_comp)
    logger.info(f"Tracking log saved to {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def replay(args):
    _dbg("replay() start")
    _dbg(f"loading episode {args.episode} from {args.dataset_dir} ...")
    actions, fps = load_episode(args.dataset_dir, args.episode)
    _dbg(f"episode loaded: {len(actions)} frames, shape={tuple(actions.shape)}")
    if args.steps is not None and args.steps > 0:
        actions = actions[:args.steps]
        logger.info(f"Truncated to first {len(actions)} frames (--steps={args.steps}).")
        _dbg(f"truncated to {len(actions)} frames")
    dt      = (1.0 / fps) / args.speed
    _dbg(f"dt = {dt:.4f}s  (speed={args.speed})")

    # DDS factory must be initialised once before any DDS publisher/subscriber.
    # Both the arm controller and the SharpaHandDDSClient use the same factory.
    need_dds = args.arm or (args.hands and args.hands_source == "dds")
    _dbg(f"need_dds={need_dds}  arm={args.arm}  hands={args.hands}  hands_source={args.hands_source}")
    if need_dds:
        _dbg(f"initialising DDS ChannelFactory on domain {args.dds_domain} (interface={args.network_interface or 'auto'}) ...")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(args.dds_domain, args.network_interface)
        logger.info(f"DDS ChannelFactory initialised on domain {args.dds_domain} "
                    f"(interface={args.network_interface or 'auto'}).")
        _dbg("DDS ChannelFactory init done")

    # ── arm ──────────────────────────────────────────────────────────────────
    arm_ctrl = None
    if args.arm:
        _dbg("importing H2_ArmController ...")
        from teleop.robot_control.robot_arm import H2_ArmController
        _dbg("H2_ArmController import done")
        logger.info(
            f"Initialising H2_ArmController (motion_mode={args.motion}, "
            f"head_pitch_home={args.head_pitch_home})..."
        )
        _dbg("constructing H2_ArmController ...")
        arm_ctrl = H2_ArmController(
            motion_mode=args.motion,
            head_pitch_home=args.head_pitch_home,
            kp_low=args.kp_low, kp_wrist=args.kp_wrist,
            kd_low=args.kd_low, kd_wrist=args.kd_wrist,
        )
        _dbg("H2_ArmController constructed; calling speed_gradual_max() ...")
        arm_ctrl.speed_gradual_max()
        _dbg("speed_gradual_max() returned")
        logger.info("H2_ArmController ready.")

    arm_ik = _make_arm_ik(bool(args.arm and args.gravity_comp))
    if args.arm and not args.gravity_comp:
        logger.info("Gravity compensation disabled (tau_ff = 0). Pass --gravity-comp to enable.")

    # ── hands ─────────────────────────────────────────────────────────────────
    left_hand  = None
    right_hand = None
    if args.hands:
        if args.hands_source == "dds":
            _dbg("connecting LEFT hand via DDS ...")
            left_hand  = connect_hand_dds("left")
            _dbg(f"LEFT hand DDS done (client={left_hand is not None})")
            _dbg("connecting RIGHT hand via DDS ...")
            right_hand = connect_hand_dds("right")
            _dbg(f"RIGHT hand DDS done (client={right_hand is not None})")
        else:
            _dbg("connecting LEFT hand via direct SDK ...")
            left_hand  = connect_hand("left",  args.hand_speed_coeff, args.hand_current_coeff)
            _dbg(f"LEFT hand SDK done (hand={left_hand is not None})")
            _dbg("connecting RIGHT hand via direct SDK ...")
            right_hand = connect_hand("right", args.hand_speed_coeff, args.hand_current_coeff)
            _dbg(f"RIGHT hand SDK done (hand={right_hand is not None})")
        _dbg("waiting 2.0s for hands to reach neutral ...")
        time.sleep(2.0)
        _dbg("hands neutral wait done")

    logger.info(f"Hardware ready.  arm={args.arm}  hands={args.hands}  repeats={args.num_repeats}")
    _dbg("hardware init complete; entering _go_home (initial)")

    # ── first home + hold ─────────────────────────────────────────────────────
    _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s)
    _dbg("initial _go_home returned")

    first_action = actions[0].numpy()

    interrupted = False
    try:
        for rep in range(args.num_repeats):
            logger.info(f"\n=== Replay {rep + 1}/{args.num_repeats} ===")
            # Smooth pre-roll to action[0]. Required when WBC is active to avoid a
            # large discontinuity when playback starts (mirrors eval_h2_groot's
            # _go_to_init_state). The H2_ArmController rate-limiter does the interp.
            _go_to_first_frame(arm_ctrl, left_hand, right_hand,
                               first_action, hold_s=args.first_frame_hold_s, arm_ik=arm_ik)
            _dbg(f"starting _run_episode rep={rep+1}/{args.num_repeats}  frames={len(actions)}")
            q_target_log, q_meas_log = _run_episode(arm_ctrl, left_hand, right_hand, actions, dt, arm_ik=arm_ik)
            _dbg(f"_run_episode rep={rep+1} returned")
            if arm_ctrl is not None:
                tag = "gc_on" if args.gravity_comp else "gc_off"
                save_path = os.path.join(
                    os.path.expanduser(args.track_log_dir),
                    f"replay_track_ep{args.episode}_{tag}_rep{rep + 1}_{time.strftime('%Y%m%d_%H%M%S')}.npz",
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
            disconnect_hand_dds(left_hand,  "left")
            disconnect_hand_dds(right_hand, "right")
        else:
            disconnect_hand(left_hand,  "left")
            disconnect_hand(right_hand, "right")
    logger.info("Replay finished.")


def parse_args():
    ap = argparse.ArgumentParser(description="Replay H2 + Sharpa episode from a LeRobot dataset.")
    ap.add_argument("--dataset-dir", required=True,
                    help="LeRobot dataset root, e.g. /home/nvidia/lerobot")
    ap.add_argument("--episode",     type=int,   default=0,   help="Episode index (default: 0)")
    ap.add_argument("--num-repeats", type=int,   default=1,   help="Number of replay repetitions (default: 1)")
    ap.add_argument("--steps",       type=int,   default=None,
                    help="Only replay the first N frames of the episode (default: all frames).")
    ap.add_argument("--first-frame-hold-s", type=int, default=3,
                    help="Seconds to hold at the first-frame posture before starting playback. "
                         "Required with --motion (WBC active) so the rate-limiter can smoothly "
                         "drive arm+hands from home-zeros to action[0] before replay begins. "
                         "Default: 3.")
    ap.add_argument("--speed",       type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    ap.add_argument("--hold-s",      type=int,   default=20,
                    help="Seconds to hold at home before/between/after the replay. Use a small "
                         "value (e.g. 3) for fast iteration, 0 to skip the final hold entirely. "
                         "Default: 20.")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arm",    dest="arm",   action="store_true",  default=True)
    g.add_argument("--no-arm", dest="arm",   action="store_false")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hands",    dest="hands", action="store_true",  default=True)
    g.add_argument("--no-hands", dest="hands", action="store_false")

    ap.add_argument("--hands-source", choices=["sdk", "dds"], default="sdk",
                    help="How to drive the Sharpa hands. 'sdk' = direct Sharpa SDK on Thor "
                         "(legacy; the bridge must be stopped). 'dds' = publish HandCmd_ to "
                         "rt/sharpa/{left,right}/cmd via the sharpa_dds_bridge running on Thor. "
                         "Default: sdk.")
    ap.add_argument("--dds-domain", type=int, default=0,
                    help="DDS domain ID for both arm and DDS-bridge hand commands (default: 0).")
    ap.add_argument("--network-interface", default=None, metavar="IFACE",
                    help="Network interface connected to the robot DDS network (e.g. eno2). "
                         "Without it the arm controller may never receive lowstate.")

    # H2 motion mode (publish to rt/arm_sdk so the WBC yields arms+head;
    # required when the robot is standing on its legs)
    ap.add_argument("--motion", action="store_true",
                    help="Enable H2 motion mode: publish on rt/arm_sdk with handover weight=1 "
                         "so the WBC yields the arm joints. Required when replaying on the "
                         "legs-active robot. Default: off (debug mode on rt/lowcmd).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gravity-comp", dest="gravity_comp", action="store_true", default=False,
                   help="Feed Pinocchio RNEA gravity torque (tau_ff = G(q)) via H2_ArmIK.compute_tauff, "
                        "same as teleop/eval. Off by default.")
    g.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false",
                   help="Send tau_ff = 0 (default).")
    ap.add_argument("--track-log-dir", default=".",
                    help="Directory for the per-replay tracking log .npz (target vs measured arm q). Default: cwd.")
    ap.add_argument("--head-pitch-home", type=float, default=0.6, metavar="RAD",
                    help="H2 head pitch held throughout replay (radians). Replay actions only "
                         "cover arms+hands, so the head stays at this fixed pose. "
                         "Range -0.523 (up) to 0.837 (down). Default: 0.6 (looking down).")

    # H2 arm PD gains (optional overrides)
    ap.add_argument("--kp-low",   type=float, default=None, help="H2 arm kp_low  (default: 150)")
    ap.add_argument("--kp-wrist", type=float, default=None, help="H2 wrist kp    (default: 50)")
    ap.add_argument("--kd-low",   type=float, default=None, help="H2 arm kd_low  (default: 10)")
    ap.add_argument("--kd-wrist", type=float, default=None, help="H2 wrist kd    (default: 3)")

    # Sharpa hand tuning
    ap.add_argument("--hand-speed-coeff",   type=float, default=0.5, help="Sharpa speed coeff 0–1 (default: 0.5)")
    ap.add_argument("--hand-current-coeff", type=float, default=0.6, help="Sharpa current coeff 0–1 (default: 0.6)")
    return ap.parse_args()


if __name__ == "__main__":
    replay(parse_args())
