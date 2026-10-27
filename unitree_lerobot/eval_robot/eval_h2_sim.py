"""
Policy inference on H2 in simulation (Isaac Sim / debug DDS domain).

Same observation/action layout as eval_h2.py but:
  - H2_ArmController(simulation_mode=True) — no velocity clipping
  - DDS domain 1 (simulation)
  - No Sharpa SDK — hand actions logged only
  - Image server expected at localhost (127.0.0.1)

Usage:
    python unitree_lerobot/eval_robot/eval_h2_sim.py \
        --policy.path /path/to/checkpoint \
        --dataset-dir /home/nvidia/lerobot

    # dataset-driven (no live policy, replay recorded actions)
    python unitree_lerobot/eval_robot/eval_h2_sim.py \
        --dataset-dir /home/nvidia/lerobot --use-dataset --episode 0
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_XR_ROOT   = os.path.expanduser("~/binliu/xr_teleoperate")

for p in [_REPO_ROOT, _XR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── action layout ─────────────────────────────────────────────────────────────
ARM_SLICE      = slice(0, 14)
LEFT_EE_SLICE  = slice(14, 36)
RIGHT_EE_SLICE = slice(36, 58)
ARM_DOF        = 14

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger = logging_mp.getLogger(__name__)


# ── dataset helpers ───────────────────────────────────────────────────────────

def _open_dataset(dataset_dir: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = os.path.expanduser(dataset_dir)
    if os.path.exists(os.path.join(dataset_dir, "meta", "info.json")):
        root, repo_id = dataset_dir, os.path.basename(dataset_dir.rstrip("/"))
    else:
        root, repo_id = os.path.dirname(dataset_dir), os.path.basename(dataset_dir.rstrip("/"))
    return LeRobotDataset(repo_id=repo_id, root=root)


# ── image helpers ─────────────────────────────────────────────────────────────

def get_observations(img_client, arm_ctrl, cam_config) -> dict:
    obs = {}

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
            img   = frame.bgr[:, :, ::-1].copy()
            obs[obs_key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    arm_q = arm_ctrl.get_current_dual_arm_q().astype(np.float32)
    # sim: hand state is unknown — use zeros
    state = np.concatenate([arm_q, np.zeros(44, dtype=np.float32)])
    obs["observation.state"] = torch.from_numpy(state)
    return obs


# ── policy helpers ────────────────────────────────────────────────────────────

def load_policy(policy_path: str, dataset, device: str):
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

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
    return policy, preprocessor, postprocessor


def predict(obs, policy, preprocessor, postprocessor, device, task=""):
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


# ── main ──────────────────────────────────────────────────────────────────────

def eval_h2_sim(args):
    device = torch.device(args.device)

    # ── DDS (sim domain = 1) ──────────────────────────────────────────────────
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1)
    from teleop.robot_control.robot_arm import H2_ArmController
    logger.info("Initialising H2_ArmController (simulation mode)...")
    arm_ctrl = H2_ArmController(simulation_mode=True)
    logger.info("H2_ArmController ready.")

    # ── image client ──────────────────────────────────────────────────────────
    from teleimager.image_client import ImageClient
    img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
    cam_config = img_client.get_cam_config()
    logger.info(f"Camera config: {list(cam_config.keys())}")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = _open_dataset(args.dataset_dir)
    task    = dataset.meta.tasks.get(0, {}).get("task", "") if hasattr(dataset.meta, "tasks") else ""

    # ── dataset-replay mode (no policy) ──────────────────────────────────────
    if args.use_dataset:
        indices = [i for i, ep in enumerate(dataset.hf_dataset["episode_index"]) if ep == args.episode]
        if not indices:
            raise ValueError(f"Episode {args.episode} not found.")
        logger.info(f"Dataset-replay mode: episode {args.episode}, {len(indices)} frames.")

        user_input = input("Press [Enter] to start: ")
        if user_input.lower() == "q":
            return

        dt = 1.0 / args.frequency
        for idx in indices:
            t0        = time.perf_counter()
            action_np = dataset[idx]["action"].numpy()
            arm_ctrl.ctrl_dual_arm(action_np[ARM_SLICE].astype(np.float64), np.zeros(ARM_DOF))
            logger.info(f"frame {idx} | arm={action_np[ARM_SLICE].round(3)} | "
                        f"left_ee={action_np[LEFT_EE_SLICE][:4].round(3)}")
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        logger.info("Dataset replay done.")
        return

    # ── live policy inference ─────────────────────────────────────────────────
    logger.info("Loading policy...")
    policy, preprocessor, postprocessor = load_policy(args.policy_path, dataset, args.device)
    logger.info(f"Policy loaded. Task: '{task}'")

    arm_ctrl.ctrl_dual_arm_go_home()
    user_input = input("\nPress [Enter] to start inference (or 'q' to quit): ")
    if user_input.lower() == "q":
        return

    logger.info(f"Starting inference at {args.frequency} Hz.")
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    dt = 1.0 / args.frequency
    try:
        while True:
            t0 = time.perf_counter()

            obs       = get_observations(img_client, arm_ctrl, cam_config)
            action_np = predict(obs, policy, preprocessor, postprocessor, device, task)

            arm_ctrl.ctrl_dual_arm(action_np[ARM_SLICE].astype(np.float64), np.zeros(ARM_DOF))
            logger.info(f"arm={action_np[ARM_SLICE].round(3)} | "
                        f"left_ee={action_np[LEFT_EE_SLICE][:4].round(3)} (sim: not sent to hands)")

            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    except KeyboardInterrupt:
        logger.info("Inference stopped by user.")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        logger.info("Done.")


def parse_args():
    ap = argparse.ArgumentParser(description="Policy inference on H2 in simulation.")
    ap.add_argument("--dataset-dir",   required=True,   help="LeRobot dataset root")
    ap.add_argument("--policy-path",   default=None,    help="Checkpoint dir (not needed with --use-dataset)")
    ap.add_argument("--img-server-ip", default="127.0.0.1", help="Image server IP (default: localhost)")
    ap.add_argument("--frequency",     type=float, default=30.0)
    ap.add_argument("--device",        default="cuda")

    ap.add_argument("--use-dataset", action="store_true",
                    help="Replay recorded dataset actions instead of running a policy")
    ap.add_argument("--episode", type=int, default=0,
                    help="Episode to replay when --use-dataset is set (default: 0)")

    # H2 PD gains (rarely needed in sim, but exposed for consistency)
    ap.add_argument("--kp-low",   type=float, default=None)
    ap.add_argument("--kp-wrist", type=float, default=None)
    ap.add_argument("--kd-low",   type=float, default=None)
    ap.add_argument("--kd-wrist", type=float, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    eval_h2_sim(parse_args())
