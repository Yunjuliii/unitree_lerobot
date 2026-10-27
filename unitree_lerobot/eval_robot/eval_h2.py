"""
Policy inference on the real H2 robot with Sharpa dexterous hands.

Observation:
  observation.state                  (58,) — 14 arm + 22 left_ee + 22 right_ee  (radians)
  observation.images.cam_high        (3, H, W) float32 in [0,1]
  observation.images.cam_left_wrist  (3, H, W) float32 in [0,1]
  observation.images.cam_right_wrist (3, H, W) float32 in [0,1]

Action (58,):
  [0:14]  — H2 dual-arm joint targets (radians)
  [14:36] — left  Sharpa hand (radians)
  [36:58] — right Sharpa hand (radians)

Usage:
    python unitree_lerobot/eval_robot/eval_h2.py \
        --policy.path /path/to/checkpoint \
        --dataset-dir /home/nvidia/lerobot \
        --img-server-ip 192.168.124.162

    # arm only (no Sharpa required)
    python unitree_lerobot/eval_robot/eval_h2.py \
        --policy.path /path/to/ckpt --no-hands \
        --img-server-ip 192.168.124.162
"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_XR_ROOT    = os.path.expanduser("~/binliu/xr_teleoperate")
_SHARPA_SDK = os.path.expanduser("~/Sharpa/SharpaWaveSDK_4.6.6/python")

for p in [_REPO_ROOT, _XR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── action layout ─────────────────────────────────────────────────────────────
ARM_SLICE       = slice(0, 14)
LEFT_EE_SLICE   = slice(14, 36)
RIGHT_EE_SLICE  = slice(36, 58)
SHARPA_DOF      = 22
ARM_DOF         = 14

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger = logging_mp.getLogger(__name__)


# ── Sharpa hand helpers ───────────────────────────────────────────────────────

def _sharpa_path():
    if _SHARPA_SDK not in sys.path:
        sys.path.insert(0, _SHARPA_SDK)


def connect_hand(side: str, speed_coeff: float = 0.5, current_coeff: float = 0.6):
    _sharpa_path()
    from sharpa import SharpaWaveManager, ControlMode, ControlSource, HandSide

    hand_side = HandSide.LEFT if side == "left" else HandSide.RIGHT
    manager   = SharpaWaveManager.get_instance()
    time.sleep(1.0)

    logger.info(f"[{side}] connecting to {hand_side.name} hand...")
    try:
        hand = manager.connect(hand_side)
    except Exception as e:
        logger.warning(f"[{side}] connect failed: {e}  — skipping")
        return None

    if hand is None:
        return None

    for fn, val in [("set_control_mode", ControlMode.POSITION), ("set_control_source", ControlSource.SDK)]:
        err = getattr(hand, fn)(val)
        if err.code != 0:
            logger.error(f"[{side}] {fn} failed: {err.message}")
            return None

    hand.set_speed_coeff(speed_coeff)
    hand.set_current_coeff(current_coeff)
    hand.start()
    hand.set_joint_position([0.0] * SHARPA_DOF, True)
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
    except Exception:
        pass
    try:
        SharpaWaveManager.get_instance().disconnect_all()
    except Exception:
        pass
    logger.info(f"[{side}] disconnected.")


def read_hand_state(hand) -> np.ndarray:
    """Read current joint angles in radians. Returns zeros on failure."""
    if hand is None:
        return np.zeros(SHARPA_DOF)
    try:
        err, angles_deg = hand.get_joint_position_degree()
        if err.code != 0:
            return np.zeros(SHARPA_DOF)
        return np.array([math.radians(a) for a in angles_deg[:SHARPA_DOF]], dtype=np.float32)
    except Exception:
        return np.zeros(SHARPA_DOF)


# ── image helpers ─────────────────────────────────────────────────────────────

def get_observations(img_client, arm_ctrl, left_hand, right_hand, cam_config) -> dict:
    """Build the observation dict for the policy."""
    obs = {}

    # images: BGR -> RGB, HWC -> CHW float32 / 255
    getter_map = {
        "observation.images.cam_high":        img_client.get_head_frame,
        "observation.images.cam_left_wrist":  img_client.get_left_wrist_frame,
        "observation.images.cam_right_wrist": img_client.get_right_wrist_frame,
    }
    cam_key_map = {
        "head_camera":        "observation.images.cam_high",
        "left_wrist_camera":  "observation.images.cam_left_wrist",
        "right_wrist_camera": "observation.images.cam_right_wrist",
    }
    for cam_name, cfg in cam_config.items():
        if not cfg.get("enable_zmq"):
            continue
        obs_key = cam_key_map.get(cam_name)
        getter  = getter_map.get(obs_key)
        if obs_key is None or getter is None:
            continue
        frame = getter()
        if frame is not None and frame.bgr is not None:
            img = frame.bgr[:, :, ::-1].copy()  # BGR→RGB
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            obs[obs_key] = img_t

    # state: [arm_q(14), left_ee(22), right_ee(22)]
    arm_q     = arm_ctrl.get_current_dual_arm_q().astype(np.float32)
    left_ee   = read_hand_state(left_hand)
    right_ee  = read_hand_state(right_hand)
    state     = np.concatenate([arm_q, left_ee, right_ee])
    obs["observation.state"] = torch.from_numpy(state)

    return obs


# ── policy helpers ────────────────────────────────────────────────────────────

def load_policy(policy_path: str, dataset_dir: str, device: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.processor.rename_processor import rename_stats

    dataset_dir = os.path.expanduser(dataset_dir)
    if os.path.exists(os.path.join(dataset_dir, "meta", "info.json")):
        root, repo_id = dataset_dir, os.path.basename(dataset_dir.rstrip("/"))
    else:
        root, repo_id = os.path.dirname(dataset_dir), os.path.basename(dataset_dir.rstrip("/"))

    dataset = LeRobotDataset(repo_id=repo_id, root=root)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset.meta.stats,
    )
    return policy, preprocessor, postprocessor, dataset


def predict(obs: dict, policy, preprocessor, postprocessor, device: torch.device, task: str = "") -> np.ndarray:
    from copy import copy
    obs = copy(obs)
    with torch.inference_mode(), (
        torch.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
    ):
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                obs[k] = v.unsqueeze(0).to(device)
        obs["task"] = task
        obs = preprocessor(obs)
        action = policy.select_action(obs)
        action = postprocessor(action)
        return action.squeeze(0).cpu().numpy()


# ── main eval loop ────────────────────────────────────────────────────────────

def eval_h2(args):
    device = torch.device(args.device)

    # ── arm ──────────────────────────────────────────────────────────────────
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0)
    from teleop.robot_control.robot_arm import H2_ArmController
    logger.info("Initialising H2_ArmController...")
    arm_ctrl = H2_ArmController(
        kp_low=args.kp_low, kp_wrist=args.kp_wrist,
        kd_low=args.kd_low, kd_wrist=args.kd_wrist,
    )
    arm_ctrl.speed_gradual_max()
    logger.info("H2_ArmController ready.")

    # ── hands ─────────────────────────────────────────────────────────────────
    left_hand  = None
    right_hand = None
    if args.hands:
        logger.info("Connecting left hand...")
        left_hand  = connect_hand("left",  args.hand_speed_coeff, args.hand_current_coeff)
        logger.info("Connecting right hand...")
        right_hand = connect_hand("right", args.hand_speed_coeff, args.hand_current_coeff)
        time.sleep(2.0)

    # ── image client ──────────────────────────────────────────────────────────
    from teleimager.image_client import ImageClient
    img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
    cam_config = img_client.get_cam_config()
    logger.info(f"Camera config: {list(cam_config.keys())}")

    # ── policy ────────────────────────────────────────────────────────────────
    logger.info("Loading policy...")
    policy, preprocessor, postprocessor, dataset = load_policy(
        args.policy_path, args.dataset_dir, args.device
    )
    task = dataset.meta.tasks.get(0, {}).get("task", "") if hasattr(dataset.meta, "tasks") else ""
    logger.info(f"Policy loaded. Task: '{task}'")

    # ── go home and wait for user ─────────────────────────────────────────────
    logger.info("Moving arm to home position...")
    arm_ctrl.ctrl_dual_arm_go_home()

    user_input = input("\nPress [Enter] to start inference (or 'q' to quit): ")
    if user_input.lower() == "q":
        logger.info("Aborted by user.")
        return

    logger.info(f"Starting inference at {args.frequency} Hz.")
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    dt = 1.0 / args.frequency
    try:
        while True:
            t0 = time.perf_counter()

            obs        = get_observations(img_client, arm_ctrl, left_hand, right_hand, cam_config)
            action_np  = predict(obs, policy, preprocessor, postprocessor, device, task)

            # execute arm
            arm_ctrl.ctrl_dual_arm(
                action_np[ARM_SLICE].astype(np.float64),
                np.zeros(ARM_DOF, dtype=np.float64),
            )

            # execute hands
            if args.hands:
                if left_hand is not None:
                    err = left_hand.set_joint_position(action_np[LEFT_EE_SLICE].tolist(), False)
                    if err.code != 0:
                        logger.warning(f"left hand set_joint_position error: {err.message}")
                if right_hand is not None:
                    err = right_hand.set_joint_position(action_np[RIGHT_EE_SLICE].tolist(), False)
                    if err.code != 0:
                        logger.warning(f"right hand set_joint_position error: {err.message}")

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        logger.info("Inference stopped by user.")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        if args.hands:
            disconnect_hand(left_hand,  "left")
            disconnect_hand(right_hand, "right")
        logger.info("Done.")


def parse_args():
    ap = argparse.ArgumentParser(description="Policy inference on H2 + Sharpa real robot.")
    ap.add_argument("--policy-path",  required=True,  help="Path to trained policy checkpoint directory")
    ap.add_argument("--dataset-dir",  required=True,  help="LeRobot dataset root (for stats/meta)")
    ap.add_argument("--img-server-ip", default="192.168.124.162", help="Image server IP")
    ap.add_argument("--frequency",    type=float, default=30.0,  help="Inference frequency Hz (default: 30)")
    ap.add_argument("--device",       default="cuda",            help="torch device (default: cuda)")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hands",    dest="hands", action="store_true",  default=True)
    g.add_argument("--no-hands", dest="hands", action="store_false")

    # H2 PD gains
    ap.add_argument("--kp-low",   type=float, default=None)
    ap.add_argument("--kp-wrist", type=float, default=None)
    ap.add_argument("--kd-low",   type=float, default=None)
    ap.add_argument("--kd-wrist", type=float, default=None)

    # Sharpa
    ap.add_argument("--hand-speed-coeff",   type=float, default=0.5)
    ap.add_argument("--hand-current-coeff", type=float, default=0.6)
    return ap.parse_args()


if __name__ == "__main__":
    eval_h2(parse_args())
