"""
H2 + Sharpa policy inference for a GR00T model trained with RELATIVE END-EFFECTOR
actions and NO state input (trocar_cotrain_rel / train_gr00t_cotrain_rel.sbatch).

Per control cycle:
    1. read the 14 arm joints  ->  FK (pinocchio, h2_with_sharpa URDF)
       -> hand_C_MC pose of each hand as [x, y, z, R00, R01, R02, R10, R11, R12]
          (GR00T XYZ_ROT6D).  Frame: PELVIS by default (--eef-frame pelvis).  The
          training data used the right-eye camera frame, but the relative pose
          T_state^-1 T_action is identical in any frame rigidly attached to the
          pelvis, so the camera transform is unnecessary; --eef-frame camera
          reproduces the training-time state values if you ever want them.
    2. send video.{high,left_wrist_view,right_wrist_view} + that pose as
       state.{left_eef,right_eef} (+ state.{left_hand,right_hand}) + language.
       The model itself never sees the state (checkpoint must carry
       exclude_state=true, see patch_groot_processor_exclude_state.py); the
       server's processor only uses it to compose the predicted RELATIVE poses
       back onto it:  T_abs[k] = T_state @ T_rel[k]   (all k anchored on the
       observation pose, i.e. "relative to the first frame").
    3. absolute poses for the horizon (same frame as the state)  ->  pelvis
       frame  ->  sequential IK (warm-started, one branch)  ->  14 joint targets
    4. execute the joint targets on the arms; hand joints are absolute targets.

The server MUST have use_relative_action=true (upstream launch_finetune sets
it): that single flag selects both the relative->absolute composition and the
relative_action statistics used to un-normalise the eef groups.  A processor
without it would return rows un-normalised with the wrong (absolute) statistics
-- not usable relative poses -- so --server-returns auto aborts when the first
chunk looks relative; --server-returns relative exists only as a debugging
override (client-side composition of whatever the server sent).

Observation keys sent (processor_config.json of the checkpoint):
    video.high               (1,1,H,W,3) uint8   <- physical head eye (--head-eye, default right)
    video.left_wrist_view    (1,1,H,W,3) uint8   } not sent with --head-only (checkpoints trained on
    video.right_wrist_view   (1,1,H,W,3) uint8   } "high" only, e.g. *_eef_camright_headonly_rel*)
    state.left_eef           (1,1,9)  float32    <- FK of hand_C_MC (pelvis frame by default)
    state.right_eef          (1,1,9)  float32
    state.left_hand          (1,1,22) float32    <- measured Sharpa joints (rad)
    state.right_hand         (1,1,22) float32
    language.annotation.human.task_description   [[str]]

Action chunk returned:  left_eef (1,32,9)  right_eef (1,32,9)  left_hand (1,32,22)  right_hand (1,32,22)
    The hand rows are ABSOLUTE 22-joint targets in both hand conventions: with ABSOLUTE hand groups
    (trocar_cotrain_rel) the model predicts them directly; with RELATIVE / NON_EEF hand groups
    (*_headonly_rel: hand[t+k] - hand[t]) the server's processor composes hand[k] = state.hand + rel[k]
    onto the hand state we sent (use_relative_action=true), exactly like the eef rows.  The hand state
    therefore has to be the real measured joints -- an all-zero read is reported as a warning.
    Launcher for the head-only / relative-hands checkpoints: run_eval_h2_groot_rel_eef_headonly.sh.

Right-hand-only / hand-state checkpoints (--arms right, run_eval_h2_groot_right_handstate.sh):
    state.right_eef + state.right_hand only; the processor zeroes right_eef in the state token
    (zero_state_keys) and feeds the real right_hand, so these checkpoints keep exclude_state=false.
    Only right_eef / right_hand come back; the left arm holds the --init-state-yaml posture and the
    left hand keeps its init posture.  --verify-checkpoint derives the expectations (exclude_state,
    zero_state_keys, state keys, video keys, eef frame) from the checkpoint itself.

GR00T server (uv env, separate terminal):
    cd ~/Isaac-GR00T
    uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path /path/to/ckpt_trocar_cotrain_rel_sd1.0_bs1024/checkpoint-XXXXX \\
        --embodiment-tag new_embodiment --port 5555

This script runs in the tv conda env (pinocchio, zmq, msgpack; no gr00t package).

Usage:
    bash unitree_lerobot/eval_robot/run_eval_h2_groot_rel_eef.sh "pick up the trocar and place it in the tray" 300
    python unitree_lerobot/eval_robot/eval_h2_groot_rel_eef.py --check --task "..."     # one observation, no server
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# Shared robot / camera / GR00T-protocol helpers (eval_h2_groot.py is left untouched).
from eval_h2_groot import (  # noqa: E402
    ARM_DOF,
    SHARPA_DOF,
    AsyncPolicyWorker,
    GR00TClient,
    _add_bt_dims,
    _frame_to_rgb,
    _go_home,
    _go_to_init_state,
    _head_right_rgb,
    _load_init_state_yaml,
    _make_image_client,
    _physical_head_side,
    _setup_hands,
    _split_head_image,
    _teardown_hands,
    _warm_cameras,
    read_hand_state,
)
from h2_eef_kinematics import (  # noqa: E402
    DEFAULT_HEAD_PITCH_RAD,
    DEFAULT_HEAD_YAW_RAD,
    EEF_DIM,
    ROBOT_ARM_SLICE,
    H2EefKinematics,
    detect_server_action_space,
    relative_chunk_to_absolute,
)

import logging_mp  # noqa: E402

logger = logging_mp.getLogger(__name__)

STATE_KEYS = ("left_eef", "right_eef", "left_hand", "right_hand")
ACTION_EEF_KEYS = {"left": "left_eef", "right": "right_eef"}
ACTION_HAND_KEYS = {"left": "left_hand", "right": "right_hand"}
LANGUAGE_KEY = "annotation.human.task_description"
EXPECTED_VIDEO_KEYS = ("high", "left_wrist_view", "right_wrist_view")
HEAD_ONLY_VIDEO_KEYS = ("high",)
WRIST_VIDEO_KEYS = ("left_wrist_view", "right_wrist_view")
H2_HEAD_PITCH_MOTOR, H2_HEAD_YAW_MOTOR = 29, 30  # H2_JointIndex.kHeadPitch / kHeadYaw


RIGHT_WRIST_ONLY_VIDEO_KEYS = ("high", "right_wrist_view")


def _expected_video_keys(args) -> tuple[str, ...]:
    """Video keys the checkpoint was trained on: head eye only (--head-only), head + right wrist only
    (--right-wrist-only), or head + both wrists (default)."""
    if args.head_only:
        return HEAD_ONLY_VIDEO_KEYS
    if getattr(args, "right_wrist_only", False):
        return RIGHT_WRIST_ONLY_VIDEO_KEYS
    return EXPECTED_VIDEO_KEYS


ALL_SIDES = ("left", "right")


def _active_sides(args) -> tuple[str, ...]:
    """Arms / hands driven by the policy: both, or the right side only (--arms right: only right_eef /
    right_hand are sent and received; the left arm and hand hold the init posture)."""
    return ("right",) if args.arms == "right" else ALL_SIDES


def _state_keys(sides) -> tuple[str, ...]:
    """state.* keys in the checkpoints' order: the eef groups first, then the hands."""
    return tuple(f"{s}_eef" for s in sides) + tuple(f"{s}_hand" for s in sides)


# ── reference state carried with every observation ───────────────────────────

@dataclass
class ObsRef:
    """What the server composed the relative chunk onto (needed for auto-detect / fallback)."""
    arm_q: np.ndarray               # (14,) measured joints at capture time
    left_eef: np.ndarray | None     # (9,) FK in the eef reference frame; None when that arm is not driven (--arms right)
    right_eef: np.ndarray | None    # (9,)
    t_obs: float


class RelEefPolicyWorker(AsyncPolicyWorker):
    """AsyncPolicyWorker whose obs_fn returns (obs, ObsRef); caches the ref with the chunk.

    get_latest() -> (chunk, t_obs, version, ref) or None.
    """

    def _loop(self):
        while not self._stop.is_set():
            if self._pause.is_set():
                if self._reset_request.is_set():
                    try:
                        self._client.reset()
                    except Exception as e:
                        logger.warning(f"async worker reset (paused) failed: {e}")
                    self._reset_request.clear()
                time.sleep(0.01)
                continue
            if self._reset_request.is_set():
                try:
                    self._client.reset()
                except Exception as e:
                    logger.warning(f"async worker reset failed: {e}")
                self._reset_request.clear()
                continue
            t_obs = time.monotonic()
            try:
                obs, ref = self._obs_fn()
                chunk, _ = self._client.get_action(obs)
            except Exception as e:
                logger.warning(f"async inference iteration failed: {e}")
                time.sleep(0.05)
                continue
            with self._cache_lock:
                self._version += 1
                self._latest = (chunk, t_obs, self._version, ref)
            self._infer_count += 1
            if self._log_every > 0 and self._infer_count % self._log_every == 0:
                logger.info(f"async worker: {self._infer_count} chunks cached (latest version={self._version})")


def _robot_dds_interface(args) -> str | None:
    """NIC for the robot DDS (rt/lowstate, rt/arm_sdk, rt/sharpa/*).  Explicit --network-interface wins;
    otherwise reuse --camera-network-interface (setup_h2.sh drives both DDS domains over the same NIC).
    CycloneDDS auto-select on a multi-NIC host (docker0, wifi, ...) often picks a NIC that never sees
    rt/lowstate, which shows up as an endless "[H2_ArmController] Waiting to subscribe dds..."."""
    if args.network_interface:
        return args.network_interface
    if args.camera_network_interface:
        logger.info(f"--network-interface not given: using the camera NIC {args.camera_network_interface!r} for the robot DDS too")
        return args.camera_network_interface
    logger.warning("--network-interface not given and no --camera-network-interface: CycloneDDS auto-selects the NIC; "
                   "if the arm controller keeps 'Waiting to subscribe dds...', pass --network-interface <NIC to the robot>")
    return None


def _init_robot_dds(args) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    iface = _robot_dds_interface(args)
    if iface:
        ChannelFactoryInitialize(args.dds_domain, iface)
    else:
        ChannelFactoryInitialize(args.dds_domain)
    logger.info(f"DDS ChannelFactory initialised on domain {args.dds_domain} (interface={iface or 'auto'}).")


# ── kinematics setup ──────────────────────────────────────────────────────────

def _read_head_joints(arm_ctrl) -> tuple[float, float]:
    q_all = arm_ctrl.get_current_motor_q()
    return float(q_all[H2_HEAD_PITCH_MOTOR]), float(q_all[H2_HEAD_YAW_MOTOR])


def _make_kinematics(args, arm_ctrl=None, *, read_head: bool = False) -> H2EefKinematics:
    """Build the kinematics.  ``read_head`` (only meaningful with --eef-frame camera) samples the
    head joints from lowstate; call it AFTER the head has been driven to --head-pitch-home
    (i.e. after _go_home / _go_to_init_state), otherwise the boot-time angles are read."""
    pitch, yaw = args.cam_head_pitch, args.cam_head_yaw
    if read_head and args.cam_head_from_robot and args.eef_frame == "camera":
        if arm_ctrl is None:
            raise RuntimeError("--cam-head-from-robot needs the arm controller (lowstate)")
        pitch, yaw = _read_head_joints(arm_ctrl)
        logger.info(f"Camera FK head joints read from the robot: pitch={pitch:.4f} yaw={yaw:.4f} rad")
    kin = H2EefKinematics(
        args.urdf,
        head_pitch_rad=pitch,
        head_yaw_rad=yaw,
        eef_frame=args.eef_frame,
        camera_eye=_physical_head_side(args.head_eye),
        ik_max_iterations=args.ik_max_iterations,
    )
    logger.info("Kinematics ready:\n  " + kin.describe().replace("\n", "\n  "))
    if args.eef_frame == "camera" and abs(args.head_pitch_home - pitch) > 0.02:
        logger.info(
            f"camera-frame FK uses head pitch/yaw {pitch:.4f} / {yaw:.4f} (the dataset's FK calibration, "
            f"--cam-head-pitch/--cam-head-yaw) while the head is commanded to {args.head_pitch_home:.3f}: expected, "
            "the calibration angles are the measured joint values of the recording, not the command; the eef math is "
            "frame-invariant anyway."
        )
    # The recordings commanded the head to DEFAULT_HEAD_PITCH_RAD (0.7); a different command changes the camera VIEW.
    # (--cam-head-pitch is the FK calibration of the dataset, e.g. 0.837 for the *_camright_* sets, not the command.)
    if abs(args.head_pitch_home - DEFAULT_HEAD_PITCH_RAD) > 0.02:
        logger.warning(
            f"--head-pitch-home {args.head_pitch_home:.3f} != {DEFAULT_HEAD_PITCH_RAD} (the head pitch COMMAND used when "
            "recording the training data): the head-camera VIEW will differ from training."
        )
    return kin


# ── optional checkpoint verification (server side config, if the path is reachable) ──

def resolve_processor_config(ckpt_dir: str) -> str:
    """Path of the processor config the GR00T server will load: Gr00tPolicy reads
    <ckpt>/processor_config.json and falls back to <ckpt>/processor/processor_config.json."""
    root = os.path.expanduser(ckpt_dir)
    candidates = [os.path.join(root, "processor_config.json"), os.path.join(root, "processor", "processor_config.json")]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"no processor config in {root} (looked for {candidates})")


def _local_gr00t_supports_zero_state_keys() -> tuple[str, bool | None]:
    """(checkout path, True/False) when a local Isaac-GR00T checkout ($GR00T_ROOT or ~/Isaac-GR00T) exists, else (path, None)."""
    root = os.environ.get("GR00T_ROOT") or os.path.expanduser("~/Isaac-GR00T")
    proc = os.path.join(root, "gr00t", "model", "gr00t_n1d7", "processing_gr00t_n1d7.py")
    if not os.path.isfile(proc):
        return root, None
    with open(proc) as f:
        return root, "zero_state_keys" in f.read()


def verify_checkpoint(ckpt_dir: str, expected_video_keys: tuple[str, ...] = EXPECTED_VIDEO_KEYS,
                      expected_state_keys: tuple[str, ...] = STATE_KEYS, eef_frame: str = "pelvis") -> dict:
    """Read the checkpoint's processor config and check the settings this client relies on.

    ``expected_video_keys``: what this client will send (all three cameras, or ("high",) with --head-only);
    ``expected_state_keys``: the state groups it will send (both sides, or right_eef/right_hand with --arms right).
    What the model may see is derived from the checkpoint itself:
      * state_dropout_prob >= 1 (no-state models): exclude_state must be true;
      * otherwise the model USES the state token: exclude_state must be false; groups listed in
        zero_state_keys are zeroed by the server (needs the zero_state_keys patch in the server's gr00t) and
        any eef group the model really sees requires --eef-frame camera (trained in the right-eye camera frame).
    Hand groups may be ABSOLUTE (trocar_cotrain_rel) or RELATIVE / NON_EEF; either way the server returns
    absolute joint targets.  Returns the settings found, incl. ``hands_rep`` and ``zero_state_keys``.
    """
    path = resolve_processor_config(ckpt_dir)
    with open(path) as f:
        cfg = json.load(f)
    kw = cfg.get("processor_kwargs", cfg)
    problems, notes = [f"config file: {path}"], []
    exclude_state = bool(kw.get("exclude_state", False))
    use_relative = bool(kw.get("use_relative_action", False))
    sd = kw.get("state_dropout_prob", None)
    no_state_model = sd is not None and float(sd) >= 1.0
    notes.append(f"exclude_state={exclude_state} use_relative_action={use_relative} state_dropout_prob={sd} -> "
                 + ("model trained WITHOUT a state token" if no_state_model else "model USES the state token"))
    problems = []
    if no_state_model and not exclude_state:
        problems.append(
            "exclude_state is not true: the server would FEED THE REAL STATE to a model trained with state "
            f"dropout {sd}. Fix: python {os.path.join(_EVAL_DIR, 'patch_groot_processor_exclude_state.py')} {ckpt_dir}"
        )
    if not no_state_model and exclude_state:
        problems.append(
            f"exclude_state is true but the model was trained WITH the state token (state_dropout_prob {sd}): the server "
            "would zero the hand state the model relies on. Undo: python "
            f"{os.path.join(_EVAL_DIR, 'patch_groot_processor_exclude_state.py')} {ckpt_dir} --unset"
        )
    if not use_relative:
        problems.append(
            "use_relative_action is false: the processor would neither compose relative->absolute NOR un-normalise "
            "the eef groups with the relative_action statistics the model was trained with -- its eef rows would be "
            "meaningless. The checkpoint was produced by a trainer that forces use_relative_action=True; do not run "
            "this one without fixing the processor config."
        )
    mcs = kw.get("modality_configs", {})
    found = False
    hands_rep: set[str] = set()
    video_keys: list[str] = []
    zero_keys: list[str] = []
    other_layouts: list[list[str]] = []
    for emb, mc in mcs.items():
        state_keys = list(mc.get("state", {}).get("modality_keys", []))
        if state_keys != list(expected_state_keys):
            if set(state_keys) <= set(STATE_KEYS) and state_keys:
                other_layouts.append(state_keys)
            continue
        found = True
        zero_keys = list(mc.get("state", {}).get("zero_state_keys") or [])
        if not no_state_model and not exclude_state:
            visible = [k for k in state_keys if k not in zero_keys]
            notes.append(f"[{emb}] state token seen by the model: {visible} (zero_state_keys={zero_keys})")
            seen_eef = [k for k in visible if k.endswith("_eef")]
            if seen_eef and eef_frame != "camera":
                problems.append(
                    f"[{emb}] the model sees the eef state {seen_eef} (not in zero_state_keys), which was trained in the "
                    "right-eye CAMERA frame: run with --eef-frame camera (pelvis-frame values would be out of distribution)"
                )
            if zero_keys:
                root, supported = _local_gr00t_supports_zero_state_keys()
                if supported is False:
                    problems.append(
                        f"[{emb}] zero_state_keys={zero_keys} but the Isaac-GR00T checkout at {root} lacks the zero_state_keys "
                        "patch: a server started from it would feed the REAL eef state to a model trained with it zeroed. "
                        "Apply isaac-ego-inpaint/cluster/gr00t_zero_state_keys.patch (or serve from a patched checkout)."
                    )
                elif supported is None:
                    notes.append(f"[{emb}] zero_state_keys={zero_keys}: no local Isaac-GR00T checkout found to check for the "
                                 "zero_state_keys patch -- the server's gr00t must have it")
                else:
                    notes.append(f"[{emb}] zero_state_keys={zero_keys}: server checkout {root} has the patch")
        video_keys = list(mc.get("video", {}).get("modality_keys", []))
        if sorted(video_keys) != sorted(expected_video_keys):
            hint = (" -> this is a head-only checkpoint: run with --head-only" if sorted(video_keys) == list(HEAD_ONLY_VIDEO_KEYS)
                    else " -> this checkpoint wants the wrist views too: drop --head-only" if "high" in video_keys
                    else "")
            problems.append(f"[{emb}] video keys {video_keys} != {list(expected_video_keys)} this client would send{hint}")
        video_horizon = len(mc.get("video", {}).get("delta_indices", [0]))
        if video_horizon != 1:
            problems.append(f"[{emb}] video delta_indices has {video_horizon} frames; this client sends a single frame (T=1)")
        lang_keys = list(mc.get("language", {}).get("modality_keys", []))
        if LANGUAGE_KEY not in lang_keys:
            problems.append(f"[{emb}] language keys {lang_keys} lack {LANGUAGE_KEY}")
        acs = mc.get("action", {}).get("action_configs") or []
        for k, ac in zip(mc.get("action", {}).get("modality_keys", []), acs):
            if k.endswith("_eef"):
                if not (ac.get("rep") == "RELATIVE" and ac.get("type") == "EEF" and ac.get("format") == "XYZ_ROT6D"):
                    problems.append(f"[{emb}] action config for {k} is {ac}, expected RELATIVE/EEF/XYZ_ROT6D")
            elif ac.get("rep") == "ABSOLUTE":
                hands_rep.add("absolute")
            elif ac.get("rep") == "RELATIVE" and ac.get("type") == "NON_EEF":
                hands_rep.add("relative")  # server composes hand[k] = state.hand + rel[k] (use_relative_action)
            else:
                problems.append(f"[{emb}] action config for {k} is {ac}, expected ABSOLUTE or RELATIVE/NON_EEF hands")
        if len(hands_rep) > 1:
            problems.append(f"[{emb}] left/right hand groups use different representations {sorted(hands_rep)}")
        action_keys = list(mc.get("action", {}).get("modality_keys", []))
        if sorted(action_keys) != sorted(state_keys):
            problems.append(f"[{emb}] action keys {action_keys} != state keys {state_keys}; this client expects one eef and "
                            "one hand group per driven side")
        horizon = len(mc.get("action", {}).get("delta_indices", []))
        notes.append(f"[{emb}] video={video_keys} state={state_keys} action_horizon={horizon} "
                     f"hands={'/'.join(sorted(hands_rep)) or 'none'}")
    if not found:
        hint = ""
        for layout in other_layouts:
            if layout == list(_state_keys(("right",))):
                hint = " -> this is a RIGHT-ONLY checkpoint: run with --arms right"
            elif layout == list(STATE_KEYS):
                hint = " -> this checkpoint drives BOTH arms: drop --arms right"
        problems.append(f"no embodiment in {path} has state keys {list(expected_state_keys)} "
                        f"(found {other_layouts or 'none of the H2 layouts'}){hint}")
    logger.info(f"checkpoint: config file {path}")
    for n in notes:
        logger.info(f"checkpoint: {n}")
    for p in problems:
        logger.error(f"checkpoint: {p}")
    if problems:
        raise RuntimeError(f"{path} is not compatible with this client:\n  - " + "\n  - ".join(problems))
    if "relative" in hands_rep:
        logger.info("checkpoint: hand groups are RELATIVE -- the server composes the returned hand targets onto the "
                    "measured hand state we send; make sure the driven Sharpa hand(s) report their joints (see warnings).")
    return {"exclude_state": exclude_state, "use_relative_action": use_relative, "path": path,
            "hands_rep": next(iter(hands_rep), None), "video_keys": video_keys,
            "state_keys": list(expected_state_keys), "zero_state_keys": zero_keys,
            "model_sees_state": (not exclude_state) and (not no_state_model)}


# ── observations ──────────────────────────────────────────────────────────────

def _capture_video(img_client, cam_config, head_eye, image_source, swap_wrist_cams,
                   video_keys: tuple[str, ...] = EXPECTED_VIDEO_KEYS) -> dict[str, np.ndarray]:
    """Grab the cameras named in ``video_keys`` (canonical names); wrist streams are not read when not wanted."""
    want_wrists = any(k in video_keys for k in WRIST_VIDEO_KEYS)
    video: dict[str, np.ndarray] = {}
    if image_source in ("dds", "ros"):
        head_rgb = _head_right_rgb(img_client, head_eye)
        if head_rgb is not None:
            video["high"] = head_rgb
        if want_wrists:
            l_wrist = img_client.get_left_wrist_frame()
            r_wrist = img_client.get_right_wrist_frame()
            if swap_wrist_cams:
                l_wrist, r_wrist = r_wrist, l_wrist
            l_rgb, r_rgb = _frame_to_rgb(l_wrist), _frame_to_rgb(r_wrist)
            if l_rgb is not None:
                video["left_wrist_view"] = l_rgb
            if r_rgb is not None:
                video["right_wrist_view"] = r_rgb
    else:  # legacy ZMQ teleimager
        head_frame = img_client.get_head_frame()
        if head_frame is not None and head_frame.bgr is not None:
            left_eye, right_eye = _split_head_image(head_frame.bgr)
            chosen = left_eye if _physical_head_side(head_eye) == "left" else right_eye
            video["high"] = chosen[:, :, ::-1].copy()
        if want_wrists:
            getters = {
                "left_wrist_view": img_client.get_right_wrist_frame if swap_wrist_cams else img_client.get_left_wrist_frame,
                "right_wrist_view": img_client.get_left_wrist_frame if swap_wrist_cams else img_client.get_right_wrist_frame,
            }
            for key, getter in getters.items():
                rgb = _frame_to_rgb(getter())
                if rgb is not None:
                    video[key] = rgb
    return {k: v for k, v in video.items() if k in video_keys}


_HAND_ZERO_WARN_T = {"left": 0.0, "right": 0.0}


def _check_hand_state(side: str, hand, values: np.ndarray) -> None:
    """Warn (throttled) when a connected hand reports all-zero joints: read_hand_state returns zeros on any
    read error, and a RELATIVE-hands model would then be composed onto a wrong reference."""
    if hand is None or np.any(values):
        return
    now = time.monotonic()
    if now - _HAND_ZERO_WARN_T[side] >= 5.0:
        _HAND_ZERO_WARN_T[side] = now
        logger.warning(f"{side} hand state reads all zeros (Sharpa DDS bridge not reporting joints, or the hand is "
                       "exactly at zero): with RELATIVE hand groups the server composes hand targets onto this state.")


def _warm_cameras_for(img_client, timeout_s: float, head_eye: str, video_keys: tuple[str, ...]) -> None:
    """Wait for the cameras that will actually be sent (head only with --head-only, otherwise all three)."""
    if any(k in video_keys for k in WRIST_VIDEO_KEYS):
        _warm_cameras(img_client, timeout_s=timeout_s, head_eye=head_eye)
        return
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if _head_right_rgb(img_client, head_eye) is not None:
            logger.info(f"Cameras warmed: ['head_{_physical_head_side(head_eye)}'] (head only)")
            return
        time.sleep(0.05)
    logger.warning(f"head_{_physical_head_side(head_eye)} camera gave no frame within {timeout_s:.0f}s "
                   "(camera publisher on Thor / --camera-network-interface?)")


def get_observations(img_client, arm_ctrl, left_hand, right_hand, cam_config, task: str, *,
                     kin: H2EefKinematics, head_eye: str, image_source: str,
                     swap_wrist_cams: bool, video_key_map: dict[str, str],
                     fallback_arm_q: np.ndarray | None = None,
                     video_poll_s: float = 0.5,
                     video_keys: tuple[str, ...] = EXPECTED_VIDEO_KEYS,
                     sides: tuple[str, ...] = ALL_SIDES) -> tuple[dict, ObsRef]:
    """Nested GR00T observation (with B=T=1 dims) + the reference state used for it.
    Only the eef / hand groups of ``sides`` are sent (--arms right -> right_eef, right_hand)."""
    deadline = time.perf_counter() + max(0.0, video_poll_s)
    while True:
        video = _capture_video(img_client, cam_config, head_eye, image_source, swap_wrist_cams, video_keys)
        if all(k in video for k in video_keys) or time.perf_counter() >= deadline:
            break
        time.sleep(0.01)
    missing = [k for k in video_keys if k not in video]
    if missing:
        raise RuntimeError(f"no camera frame for {missing} (got {sorted(video)}); cameras not publishing?")
    video = {video_key_map.get(k, k): v for k, v in video.items()}

    if arm_ctrl is not None:
        arm_q = arm_ctrl.get_current_dual_arm_q().astype(np.float64)
    else:
        arm_q = np.zeros(ARM_DOF) if fallback_arm_q is None else np.asarray(fallback_arm_q, dtype=np.float64)
    left_eef, right_eef = kin.fk_ref(arm_q)
    eef = {"left": left_eef, "right": right_eef}
    hands = {"left": left_hand, "right": right_hand}
    state: dict[str, np.ndarray] = {}
    for side in sides:  # checkpoint key order: eef groups, then hands (see _state_keys)
        state[f"{side}_eef"] = eef[side].astype(np.float32)
    for side in sides:
        hand_q = read_hand_state(hands[side]).astype(np.float32)
        _check_hand_state(side, hands[side], hand_q)
        state[f"{side}_hand"] = hand_q
    obs = {"video": video, "state": state, "language": {LANGUAGE_KEY: task}}
    ref = ObsRef(arm_q=arm_q,
                 left_eef=left_eef.astype(np.float64) if "left" in sides else None,
                 right_eef=right_eef.astype(np.float64) if "right" in sides else None,
                 t_obs=time.monotonic())
    return _add_bt_dims(obs), ref


def _video_key_map(args) -> dict[str, str]:
    return {"high": args.video_key_head, "left_wrist_view": args.video_key_wrist_left,
            "right_wrist_view": args.video_key_wrist_right}


# ── chunk -> joint trajectory ─────────────────────────────────────────────────

def _chunk_len(chunk: dict, fallback: int) -> int:
    for k in ACTION_EEF_KEYS.values():
        if k in chunk:
            return int(np.asarray(chunk[k]).shape[1])
    return fallback


def _eef_rows(chunk: dict, side: str) -> np.ndarray:
    key = ACTION_EEF_KEYS[side]
    if key not in chunk:
        raise KeyError(f"action chunk has no {key!r}; keys={sorted(chunk)}. Wrong checkpoint / embodiment?")
    rows = np.asarray(chunk[key], dtype=np.float64)
    if rows.ndim == 3:
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] != EEF_DIM:
        raise ValueError(f"{key} has shape {np.asarray(chunk[key]).shape}, expected (1, T, 9)")
    if not np.isfinite(rows).all():
        raise ValueError(f"{key} contains non-finite values")
    return rows


def resolve_server_mode(chunk: dict, ref: ObsRef, requested: str, sides: tuple[str, ...] = ALL_SIDES) -> str:
    """'absolute' | 'relative' -- what the server's eef rows are.

    A correctly configured GR00T server (use_relative_action=true) returns ABSOLUTE
    poses in the frame of the state it received.  Rows that look RELATIVE mean the
    processor lacks use_relative_action, in which case they were also un-normalised
    with the absolute instead of the relative_action statistics and are NOT valid
    relative poses -- so 'auto' refuses to run on them.
    """
    if requested in ("absolute", "relative"):
        if requested == "relative":
            logger.warning("--server-returns relative: composing the server rows client-side. A stock GR00T processor "
                           "cannot produce valid raw relative poses (wrong statistics); debugging use only.")
        return requested
    votes, reasons = [], []
    for side in sides:
        mode, reason = detect_server_action_space(_eef_rows(chunk, side), getattr(ref, f"{side}_eef"))
        votes.append(mode)
        reasons.append(f"{side}: {mode} ({reason})")
    if all(v == "absolute" for v in votes):
        logger.info(f"server eef output detected as ABSOLUTE  [{' | '.join(reasons)}]")
        return "absolute"
    if all(v == "relative" for v in votes):
        raise RuntimeError(
            "The server returned RELATIVE eef rows (" + " | ".join(reasons) + "). Its processor has "
            "use_relative_action=false, so these rows were un-normalised with the wrong statistics and cannot be "
            "executed. Fix the checkpoint's processor_config.json (use_relative_action: true, exclude_state: true) "
            "and restart the server. --server-returns relative overrides this check for debugging only."
        )
    raise RuntimeError(
        "Cannot tell whether the server returns absolute or relative eef poses: "
        + " | ".join(reasons) + ". Check the server config; --server-returns absolute forces the normal path."
    )


def absolute_ref_rows(chunk: dict, ref: ObsRef, mode: str, sides: tuple[str, ...] = ALL_SIDES) -> dict[str, np.ndarray]:
    """Absolute eef rows (T, 9) per driven side in the reference frame of the state we sent."""
    out = {}
    for side in sides:
        rows = _eef_rows(chunk, side)
        out[side] = relative_chunk_to_absolute(getattr(ref, f"{side}_eef"), rows) if mode == "relative" else rows
    return out


def report_bad_steps(res, max_pos_m: float, max_rot_rad: float, held: bool) -> None:
    """Log the chunk steps whose IK residual exceeds the limits (held or executed as solved)."""
    bad = (res.pos_err_m > max_pos_m) | (res.rot_err_rad > max_rot_rad)
    if not bad.any():
        return
    steps = np.where(bad.any(axis=1))[0]
    arms = "/".join(a for a, m in (("L", bad[:, 0].any()), ("R", bad[:, 1].any())) if m)
    logger.warning(
        f"IK residual above --ik-max-pos-err-m {max_pos_m * 1e3:.0f} mm / --ik-max-rot-err-deg "
        f"{np.degrees(max_rot_rad):.0f} deg on steps {steps.tolist()} (arm {arms}; worst pos "
        f"{res.pos_err_m.max() * 1e3:.1f} mm, rot {np.degrees(res.rot_err_rad.max()):.1f} deg)"
        + (f" -> that arm held its previous joints on {int(res.held.sum())} arm-step(s)" if held
           else " -> executing the IK result anyway (--ik-hold-on-error to hold instead)")
    )


# ── check mode ────────────────────────────────────────────────────────────────

DATASET_VIDEO_KEYS = {
    "high": "observation.images.head_right",
    "left_wrist_view": "observation.images.wrist_left",
    "right_wrist_view": "observation.images.wrist_right",
}


def dump_reference_frames(dataset_dir: str, episode: int, video_keys: tuple[str, ...] = EXPECTED_VIDEO_KEYS) -> None:
    """Save the first frame of each camera of one training episode next to the live snapshots,
    so the operator can see whether the live streams are labelled like the training data
    (trocar_cotrain_rel's teleop half carries the wrist labels of the recording; if the live
    left/right look swapped relative to it, run with --swap-wrist-cams).  The policy-key ->
    dataset-column mapping is read from meta/modality.json when present."""
    import cv2
    root = os.path.expanduser(dataset_dir)
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    chunk = episode // int(info.get("chunks_size", 1000))
    mapping = dict(DATASET_VIDEO_KEYS)
    modality_path = os.path.join(root, "meta", "modality.json")
    if os.path.isfile(modality_path):
        for key, spec in json.load(open(modality_path)).get("video", {}).items():
            if isinstance(spec, dict) and spec.get("original_key"):
                mapping[key] = spec["original_key"]
    for key in video_keys:
        ds_key = mapping.get(key)
        if ds_key is None:
            logger.warning(f"  [dataset {key}] no dataset column known for this key (meta/modality.json?)")
            continue
        rel = info["video_path"].format(episode_chunk=chunk, video_key=ds_key, episode_index=episode)
        path = os.path.join(root, rel)
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            logger.warning(f"  [dataset {key}] could not read {path}")
            continue
        out = f"/tmp/check_rel_eef_dataset_{key}.png"
        cv2.imwrite(out, frame)
        logger.info(f"  [dataset {key}] episode {episode} first frame -> {out}   (compare with /tmp/check_rel_eef_{key}.png)")


def check_observations(args):
    import cv2

    video_keys = _expected_video_keys(args)
    sides = _active_sides(args)
    if args.verify_checkpoint:
        verify_checkpoint(args.verify_checkpoint, video_keys, _state_keys(sides), args.eef_frame)

    need_dds = args.arm or (args.hands and args.hands_source == "dds")
    if need_dds:
        _init_robot_dds(args)

    left_hand, right_hand = _setup_hands(args)
    arm_ctrl = None
    if args.arm:
        from teleop.robot_control.robot_arm import H2_ArmController
        arm_ctrl = H2_ArmController(motion_mode=args.motion, head_pitch_home=args.head_pitch_home,
                                    kp_low=args.kp_low, kp_wrist=args.kp_wrist, kd_low=args.kd_low, kd_wrist=args.kd_wrist)
    kin = _make_kinematics(args, arm_ctrl)

    img_client, cam_config = _make_image_client(args)
    _warm_cameras_for(img_client, args.cam_warmup_s, args.head_eye, video_keys)
    obs, ref = get_observations(img_client, arm_ctrl, left_hand, right_hand, cam_config, args.task,
                                kin=kin, head_eye=args.head_eye, image_source=args.image_source,
                                swap_wrist_cams=args.swap_wrist_cams, video_key_map=_video_key_map(args),
                                video_keys=video_keys, sides=sides)

    def _print(d, prefix=""):
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _print(v, path)
            elif isinstance(v, np.ndarray):
                logger.info(f"  {path:45s} shape={str(v.shape):16s} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}")
            else:
                logger.info(f"  {path:45s} value={v!r}")

    logger.info("\n=== Observation (as sent to the server) ===")
    _print(obs)
    logger.info(f"  measured arm q (rad): {np.round(ref.arm_q, 4).tolist()}")
    for side in sides:
        logger.info(f"  FK {side}_eef ({kin.eef_frame} frame, xyz+rot6d): {np.round(getattr(ref, f'{side}_eef'), 4).tolist()}")
    if len(sides) < len(ALL_SIDES):
        logger.info(f"  --arms {args.arms}: only {list(sides)} is sent / driven; the other arm and hand hold the init posture")
    # IK self-consistency on the current pose (held arm: None)
    res = kin.solve_ref_chunk(None if ref.left_eef is None else ref.left_eef[None],
                              None if ref.right_eef is None else ref.right_eef[None], ref.arm_q)
    logger.info(f"  IK round trip on the current pose: {res.summary()}; "
                f"max |q_ik - q| = {np.degrees(np.abs(res.q_traj[0] - ref.arm_q)).max():.3f} deg")
    for key, arr in obs.get("video", {}).items():
        save_path = f"/tmp/check_rel_eef_{key}.png"
        cv2.imwrite(save_path, arr[0, 0][:, :, ::-1])
        logger.info(f"  [{key}] saved -> {save_path}")
    if args.reference_dataset:
        dump_reference_frames(args.reference_dataset, args.reference_episode, video_keys)
        if any(k in video_keys for k in WRIST_VIDEO_KEYS):
            logger.info("  If the live left/right wrist views are mirrored w.r.t. the dataset frames, add --swap-wrist-cams.")
        else:
            logger.info("  Compare the live head view with the dataset frame: same eye (--head-eye) and similar head pitch?")
    _teardown_hands(args, left_hand, right_hand)
    try:
        img_client.close()
    except Exception:
        pass
    logger.info("Check complete.")


# ── main eval loop ────────────────────────────────────────────────────────────

def eval_h2_groot_rel_eef(args):
    if args.check:
        check_observations(args)
        return

    video_keys = _expected_video_keys(args)
    sides = _active_sides(args)
    if args.verify_checkpoint:
        verify_checkpoint(args.verify_checkpoint, video_keys, _state_keys(sides), args.eef_frame)
    else:
        logger.warning(
            "Checkpoint NOT verified (no --verify-checkpoint / REL_EEF_CKPT). The server gives no signal about it, so "
            "make sure its processor_config.json matches the model: exclude_state=true for the no-state models "
            "(state_dropout 1.0), exclude_state=false (+ zero_state_keys for the eef groups) for the hand-state models, "
            "use_relative_action=true, and video/state keys = what this client sends "
            f"({list(video_keys)} / {list(_state_keys(sides))})."
        )

    if args.arm or (args.hands and args.hands_source == "dds"):
        _init_robot_dds(args)

    left_hand, right_hand = _setup_hands(args)

    arm_ctrl = None
    if args.arm:
        from teleop.robot_control.robot_arm import H2_ArmController
        logger.info(f"Initialising H2_ArmController (motion_mode={args.motion}, head_pitch_home={args.head_pitch_home})...")
        arm_ctrl = H2_ArmController(motion_mode=args.motion, head_pitch_home=args.head_pitch_home,
                                    kp_low=args.kp_low, kp_wrist=args.kp_wrist, kd_low=args.kd_low, kd_wrist=args.kd_wrist)
        arm_ctrl.speed_gradual_max()
        logger.info("H2_ArmController ready.")
    else:
        logger.warning("--no-arm: state.*_eef will be the FK of the init posture / zeros; arms are not commanded.")

    kin = _make_kinematics(args, arm_ctrl)  # head joints re-read below once the head has reached its home pitch
    tauff_provider = kin if (args.arm and args.gravity_comp) else None
    logger.info("Gravity compensation " + ("enabled (pinocchio RNEA on the URDF incl. hands)." if tauff_provider else "disabled (tau_ff = 0)."))

    init_state = _load_init_state_yaml(args.init_state_yaml) if args.init_state_yaml else None
    fallback_arm_q = None
    if init_state is not None:
        fallback_arm_q = np.concatenate([
            np.asarray(init_state.get("left_arm", {}).get("qpos", [0.0] * 7), dtype=np.float64),
            np.asarray(init_state.get("right_arm", {}).get("qpos", [0.0] * 7), dtype=np.float64),
        ])

    hold_q = None  # joints of the arm(s) the policy does not drive (--arms right): the init posture when known
    if len(sides) < len(ALL_SIDES):
        held = [s for s in ALL_SIDES if s not in sides]
        if fallback_arm_q is not None:
            hold_q = fallback_arm_q.copy()
            logger.info(f"--arms {args.arms}: the {held} arm holds the --init-state-yaml posture on every step; "
                        f"the {held} hand keeps its init posture (not commanded during the episode).")
        else:
            logger.info(f"--arms {args.arms}: no --init-state-yaml -> the {held} arm holds the joints it has at the first "
                        f"chunk of each episode; the {held} hand is not commanded during the episode.")

    _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s, arm_ik=tauff_provider)
    if init_state is not None:
        _go_to_init_state(arm_ctrl, left_hand, right_hand, init_state, hold_s=args.init_state_hold_s, arm_ik=tauff_provider)
    if args.cam_head_from_robot and args.eef_frame == "camera" and arm_ctrl is not None:
        kin = _make_kinematics(args, arm_ctrl, read_head=True)  # head is at --head-pitch-home now
        tauff_provider = kin if (args.arm and args.gravity_comp) else None

    img_client, cam_config = _make_image_client(args)
    logger.info(f"Camera config: {list(cam_config.keys())}; video keys sent: {list(video_keys)}; "
                f"video.high <- physical {_physical_head_side(args.head_eye)} eye")
    _warm_cameras_for(img_client, args.cam_warmup_s, args.head_eye, video_keys)

    logger.info(f"Connecting to GR00T server at {args.policy_host}:{args.policy_port} ...")
    client = GR00TClient(host=args.policy_host, port=args.policy_port, timeout_ms=args.timeout_ms)
    for attempt in range(30):
        if client.ping():
            logger.info("GR00T server: connected.")
            break
        logger.info(f"  waiting for server... ({attempt + 1}/30)")
        time.sleep(2.0)
    else:
        raise RuntimeError("Could not reach GR00T server after 60s. Is it running?")

    log_dir = None
    if args.log_dir:
        log_dir = os.path.join(os.path.expanduser(args.log_dir), time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "run_meta.json"), "w") as f:
            json.dump({"args": vars(args), "kinematics": kin.describe(), "eef_frame": kin.eef_frame,
                       "video_keys": list(video_keys), "sides": list(sides), "T_pelvis_ref": kin.T_pelvis_ref.tolist(),
                       "T_pelvis_camera": kin.T_pelvis_camera.tolist()}, f, indent=2, default=str)
        logger.info(f"Per-chunk debug dumps -> {log_dir}")

    logger.info(f"Inference config: {args.frequency} Hz, action_horizon={args.action_horizon}, steps={args.steps}, "
                f"task='{args.task}', async={args.async_policy}, eef_frame={args.eef_frame}, head_only={args.head_only}, "
                f"arms={args.arms}, server_returns={args.server_returns}, "
                f"max_joint_step={np.degrees(args.max_joint_step_rad):.0f} deg/step")

    def _obs_fn():
        return get_observations(img_client, arm_ctrl, left_hand, right_hand, cam_config, args.task,
                                kin=kin, head_eye=args.head_eye, image_source=args.image_source,
                                swap_wrist_cams=args.swap_wrist_cams, video_key_map=_video_key_map(args),
                                fallback_arm_q=fallback_arm_q, video_keys=video_keys, sides=sides)

    dt = 1.0 / args.frequency
    episode = 0
    server_mode = args.server_returns  # resolved on the first chunk when 'auto'
    chunk_counter = 0

    worker = None
    if args.async_policy:
        worker = RelEefPolicyWorker(client, _obs_fn)
        worker.start()
        logger.info("Async policy worker started (paused).")

    try:
        # Inside the try so that 'q' / Ctrl+C here still runs the teardown below (hands are already
        # at the init posture and the camera client is open).
        if input("\nPress [Enter] to start inference (or 'q' to quit): ").lower() == "q":
            return

        while True:
            episode += 1
            if worker is not None:
                worker.pause()
                worker.request_reset()
                worker.resume()
            else:
                client.reset()
            logger.info(f"\n=== Episode {episode} ===")

            smooth_alpha = float(args.smooth_alpha)
            boundary_alpha = float(args.boundary_smooth_alpha)
            boundary_total = max(0, int(args.boundary_smooth_steps))
            blend_total = max(0, int(args.boundary_blend_steps))
            blend_alpha = max(1e-6, float(args.boundary_blend_alpha))

            prev_arm_q = None            # last commanded 14 joints (EMA state + IK seed)
            prev_hand = {"left": None, "right": None}
            prev_traj = None             # previous chunk's IK trajectory (full chunk, index = chunk step)
            prev_chunk = None
            prev_next_t = 0
            last_version = -1
            steps_executed = 0

            while steps_executed < args.steps:
                # ── fetch a chunk (+ the reference state it was composed on) ──
                if worker is not None:
                    waited = time.monotonic()
                    while True:
                        latest = worker.get_latest()
                        if latest is not None and latest[2] > last_version:
                            break
                        if time.monotonic() - waited > 30.0:
                            raise RuntimeError("Async worker produced no chunk in 30s.")
                        time.sleep(args.async_cache_poll_s)
                    chunk, t_obs, last_version, ref = latest
                else:
                    obs, ref = _obs_fn()
                    t_obs = time.monotonic()
                    chunk, _ = client.get_action(obs)
                    logger.info(f"inference {(time.monotonic() - t_obs) * 1e3:.0f} ms")
                chunk_len = _chunk_len(chunk, args.action_horizon)

                if server_mode == "auto":
                    server_mode = resolve_server_mode(chunk, ref, "auto", sides)

                # ── eef rows -> pelvis targets -> IK over the WHOLE chunk (~0.1 ms per step) ──
                # Solving everything first lets the async start_step below account for the IK time too.
                # An arm the policy does not drive gets no targets (None) and keeps its seed joints = hold_q.
                abs_rows = absolute_ref_rows(chunk, ref, server_mode, sides)
                q_seed = (prev_arm_q if prev_arm_q is not None else ref.arm_q).copy()
                if hold_q is not None:
                    for side in ALL_SIDES:
                        if side not in sides:
                            q_seed[ROBOT_ARM_SLICE[side]] = hold_q[ROBOT_ARM_SLICE[side]]
                res = kin.solve_ref_chunk(
                    abs_rows.get("left"), abs_rows.get("right"), q_seed,
                    max_joint_step_rad=args.max_joint_step_rad,
                    hold_pos_err_m=args.ik_max_pos_err_m if args.ik_hold_on_error else None,
                    hold_rot_err_rad=np.deg2rad(args.ik_max_rot_err_deg) if args.ik_hold_on_error else None,
                )
                traj = res.q_traj
                chunk_counter += 1
                report_bad_steps(res, args.ik_max_pos_err_m, np.deg2rad(args.ik_max_rot_err_deg), args.ik_hold_on_error)
                if res.clamped.any():
                    logger.warning(f"joint-step guard clamped {int(res.clamped.sum())} step(s) "
                                   f"(> {np.degrees(args.max_joint_step_rad):.0f} deg between consecutive targets)")

                # ── which chunk steps are still in the future? ──
                # async: skip the steps that elapsed during observation + inference + IK; a chunk older than its
                # whole horizon is dropped (its reference pose is stale) and we wait for the next one.
                # sync: execute from step 0, like eval_h2_groot.py.
                age_s = time.monotonic() - t_obs
                start_step = max(int(age_s / dt), 0) if worker is not None else 0
                horizon = min(args.action_horizon, chunk_len - start_step, args.steps - steps_executed)
                if horizon <= 0:
                    logger.warning(f"chunk {chunk_counter} dropped: {age_s * 1e3:.0f} ms old "
                                   f"(start_step={start_step} >= chunk_len={chunk_len}); waiting for a fresh one")
                    continue
                first_move = float(np.degrees(np.abs(traj[start_step] - q_seed)).max())
                logger.info(f"chunk {chunk_counter}: age {age_s * 1e3:.0f} ms, start_step={start_step} horizon={horizon} | "
                            f"{res.summary()} | first executed step {first_move:.1f} deg from the last command")
                if first_move > np.degrees(args.max_joint_step_rad):
                    logger.warning(f"first executed step jumps {first_move:.1f} deg from the last command "
                                   f"(> {np.degrees(args.max_joint_step_rad):.0f} deg): stale chunk or the model disagrees "
                                   "with the previous chunk; the arm controller's rate limit smooths it")

                if log_dir is not None:
                    dump = dict(
                        arm_q_measured=ref.arm_q, q_seed=q_seed, q_traj=traj, ik_pos_err_m=res.pos_err_m,
                        ik_rot_err_rad=res.rot_err_rad, ik_held=res.held, ik_clamped=res.clamped,
                        start_step=start_step, horizon=horizon, server_mode=server_mode, eef_frame=kin.eef_frame,
                        sides=np.array(sides), t_obs=ref.t_obs, t_exec=time.monotonic(),
                    )
                    for side in sides:
                        dump[f"state_{side}_eef"] = getattr(ref, f"{side}_eef")
                        dump[f"server_{side}_eef"] = _eef_rows(chunk, side)
                        dump[f"abs_{side}_eef"] = abs_rows[side]
                        dump[f"{side}_hand"] = np.asarray(chunk.get(ACTION_HAND_KEYS[side], np.zeros((1, 0))))
                    np.savez_compressed(os.path.join(log_dir, f"ep{episode:02d}_chunk{chunk_counter:04d}.npz"), **dump)

                # ── cross-chunk blend window (uses the previous chunk's unexecuted IK steps) ──
                if blend_total > 0 and prev_traj is not None:
                    old_remaining = max(0, prev_traj.shape[0] - prev_next_t)
                    blend_window = min(blend_total, old_remaining, horizon)
                else:
                    blend_window = 0
                boundary_remaining = boundary_total

                # ── execute ──
                for offset in range(horizon):
                    t = start_step + offset
                    t0 = time.perf_counter()
                    if boundary_remaining > 0:
                        effective_alpha = boundary_alpha
                        boundary_remaining -= 1
                    else:
                        effective_alpha = smooth_alpha

                    raw_q = traj[t].copy()
                    if offset < blend_window:
                        old_t = prev_next_t + offset
                        old_q = prev_traj[old_t]
                        if np.isfinite(old_q).all():
                            w_old = math.exp(-blend_alpha * offset / blend_window)
                            raw_q = w_old * old_q + (1.0 - w_old) * raw_q
                    if effective_alpha < 1.0 and prev_arm_q is not None:
                        arm_q_target = effective_alpha * raw_q + (1.0 - effective_alpha) * prev_arm_q
                    else:
                        arm_q_target = raw_q
                    prev_arm_q = arm_q_target
                    if arm_ctrl is not None:
                        tau = tauff_provider.compute_tauff(arm_q_target) if tauff_provider else np.zeros(ARM_DOF)
                        arm_ctrl.ctrl_dual_arm(np.asarray(arm_q_target, dtype=np.float64), tau)

                    if args.hands:
                        for side, hand in (("left", left_hand), ("right", right_hand)):
                            key = ACTION_HAND_KEYS[side]
                            if hand is None or key not in chunk:
                                continue
                            raw_h = np.asarray(chunk[key][0][t], dtype=np.float64)
                            if offset < blend_window and prev_chunk is not None and key in prev_chunk:
                                old_t = prev_next_t + offset
                                w_new = (offset + 1) / float(blend_window)
                                raw_h = w_new * raw_h + (1.0 - w_new) * np.asarray(prev_chunk[key][0][old_t], dtype=np.float64)
                            if effective_alpha < 1.0 and prev_hand[side] is not None:
                                h = effective_alpha * raw_h + (1.0 - effective_alpha) * prev_hand[side]
                            else:
                                h = raw_h
                            prev_hand[side] = h
                            err = hand.set_joint_position(h.tolist(), False)
                            if err is not None and getattr(err, "code", 0) != 0:
                                logger.warning(f"{side} hand error t={t}: {getattr(err, 'message', err)}")

                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                    steps_executed += 1

                logger.info(f"chunk done ({horizon} steps from start_step={start_step}), total={steps_executed}/{args.steps}")
                prev_chunk = chunk
                prev_traj = traj
                prev_next_t = start_step + horizon

            if worker is not None:
                worker.pause()
            logger.info(f"Episode {episode} complete ({args.steps} steps). Going home...")
            _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s, arm_ik=tauff_provider)
            if init_state is not None:
                _go_to_init_state(arm_ctrl, left_hand, right_hand, init_state, hold_s=args.init_state_hold_s, arm_ik=tauff_provider)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        if worker is not None:
            try:
                worker.stop()
            except Exception as e:
                logger.warning(f"async worker stop failed: {e}")
        _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s, arm_ik=tauff_provider)
        _teardown_hands(args, left_hand, right_hand)
        try:
            img_client.close()
        except Exception:
            pass
        logger.info("Done.")


def parse_args():
    ap = argparse.ArgumentParser(
        description="H2 + Sharpa eval for a GR00T RELATIVE-EEF / no-state policy (hand_C_MC poses, pelvis frame by default).")
    ap.add_argument("--check", action="store_true", help="Build one observation (FK state + cameras), print it, exit. No server needed.")
    ap.add_argument("--policy-host", default="localhost")
    ap.add_argument("--policy-port", type=int, default=5555)
    ap.add_argument("--timeout-ms", type=int, default=15000)
    ap.add_argument("--task", default="", help="Language instruction (annotation.human.task_description)")
    ap.add_argument("--img-server-ip", default="192.168.124.162")
    ap.add_argument("--frequency", type=float, default=30.0, help="Control rate Hz (= dataset fps, one chunk step per tick)")
    ap.add_argument("--action-horizon", type=int, default=16, help="Chunk steps executed before re-querying (chunk has 32)")
    ap.add_argument("--steps", type=int, default=30, help="Control steps per episode")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hands", dest="hands", action="store_true", default=True)
    g.add_argument("--no-hands", dest="hands", action="store_false")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arm", dest="arm", action="store_true", default=True)
    g.add_argument("--no-arm", dest="arm", action="store_false")

    # cameras
    ap.add_argument("--image-source", choices=["dds", "ros", "zmq"], default="dds")
    ap.add_argument("--camera-dds-domain", type=int, default=10)
    ap.add_argument("--camera-network-interface", default=None)
    ap.add_argument("--head-eye", choices=["left", "right", "both"], default="right",
                    help="Physical head eye that fills video.high (and, with --eef-frame camera, the eef frame). "
                         "trocar_cotrain_rel is right-eye: keep 'right'.")
    ap.add_argument("--cam-warmup-s", type=float, default=5.0)
    ap.add_argument("--head-only", action="store_true",
                    help="Send only video.high (the head eye): for checkpoints trained without wrist views "
                         "(*_eef_camright_headonly_rel*, video modality_keys == ['high']). The wrist cameras are "
                         "neither read nor required. --verify-checkpoint cross-checks this against the processor config.")
    ap.add_argument("--arms", choices=["both", "right"], default="both",
                    help="Arms / hands driven by the policy. 'right': send only state.right_eef / right_hand and execute "
                         "only action.right_eef / right_hand (checkpoints trained on the *_right datasets); the left arm "
                         "holds the --init-state-yaml posture (else the joints it has at episode start) and the left hand "
                         "keeps its init posture.")
    ap.add_argument("--right-wrist-only", action="store_true",
                    help="Send video.high + video.right_wrist_view but NOT video.left_wrist_view: for checkpoints "
                         "trained on ['high', 'right_wrist_view'] only (video modality_keys), typically paired with "
                         "--arms right. Ignored with --head-only. --verify-checkpoint cross-checks this against the "
                         "processor config.")
    ap.add_argument("--swap-wrist-cams", action="store_true")
    ap.add_argument("--video-key-head", default="high", metavar="KEY")
    ap.add_argument("--video-key-wrist-left", default="left_wrist_view", metavar="KEY")
    ap.add_argument("--video-key-wrist-right", default="right_wrist_view", metavar="KEY")

    # robot
    ap.add_argument("--hands-source", choices=["dds", "sdk"], default="dds")
    ap.add_argument("--dds-domain", type=int, default=0)
    ap.add_argument("--network-interface", default=None, metavar="IFACE")
    ap.add_argument("--motion", action="store_true", help="Publish on rt/arm_sdk (robot standing).")
    ap.add_argument("--head-pitch-home", type=float, default=DEFAULT_HEAD_PITCH_RAD, metavar="RAD",
                    help="Head pitch COMMAND held during inference; the recordings commanded 0.7 rad.")
    ap.add_argument("--hold-s", type=int, default=20)
    ap.add_argument("--init-state-yaml", type=str, default=None)
    ap.add_argument("--init-state-hold-s", type=int, default=3)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gravity-comp", dest="gravity_comp", action="store_true", default=False,
                   help="tau_ff = G(q) from pinocchio RNEA on the h2_with_sharpa URDF (hands included).")
    g.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false")
    ap.add_argument("--kp-low", type=float, default=None)
    ap.add_argument("--kp-wrist", type=float, default=None)
    ap.add_argument("--kd-low", type=float, default=None)
    ap.add_argument("--kd-wrist", type=float, default=None)
    ap.add_argument("--hand-speed-coeff", type=float, default=0.5)
    ap.add_argument("--hand-current-coeff", type=float, default=0.6)

    # kinematics / eef
    ap.add_argument("--urdf", default=None, help="H2_with_sharpa_hands.urdf (default: $H2_IK_URDF or ~/xr_teleoperate/h2_with_sharpa/urdf/...)")
    ap.add_argument("--eef-frame", choices=["pelvis", "camera"], default="pelvis",
                    help="Frame of the 9-dim eef state/targets. 'pelvis' (default): FK straight in the pelvis frame; "
                         "the relative poses are frame-invariant so the result is identical. 'camera': right-eye camera "
                         "frame exactly as in the training data (only matters if the model could see the state).")
    ap.add_argument("--cam-head-pitch", type=float, default=DEFAULT_HEAD_PITCH_RAD, metavar="RAD",
                    help="[--eef-frame camera] head pitch the dataset builder used for the camera-frame FK of the "
                         "training eef poses (measured joint value, not the head command): 0.7 for trocar_test / "
                         "trocar_cotrain_rel, 0.837 for the *_eef_camright_* sets.")
    ap.add_argument("--cam-head-yaw", type=float, default=DEFAULT_HEAD_YAW_RAD, metavar="RAD",
                    help="[--eef-frame camera] head yaw of that FK calibration: 0.095 for trocar_test / "
                         "trocar_cotrain_rel, -0.0125 for the *_eef_camright_* sets.")
    ap.add_argument("--cam-head-from-robot", action="store_true",
                    help="[--eef-frame camera] read the head joints from rt/lowstate once at start instead.")
    ap.add_argument("--server-returns", choices=["auto", "absolute", "relative"], default="auto",
                    help="What the server's *_eef rows are. A correct GR00T processor (use_relative_action=true) returns "
                         "ABSOLUTE poses in the frame of the state it was given; 'auto' checks the first chunk and ABORTS "
                         "if the rows look relative (misconfigured processor: wrong statistics). 'relative' = compose "
                         "client-side anyway, debugging only.")
    ap.add_argument("--max-joint-step-rad", type=float, default=0.35,
                    help="IK guard: max joint change between consecutive chunk steps (branch-flip protection).")
    ap.add_argument("--ik-max-pos-err-m", type=float, default=0.03)
    ap.add_argument("--ik-max-rot-err-deg", type=float, default=20.0)
    ap.add_argument("--ik-hold-on-error", action="store_true",
                    help="Hold the previous joint target on steps whose IK residual exceeds the limits above.")
    ap.add_argument("--ik-max-iterations", type=int, default=40)
    ap.add_argument("--verify-checkpoint", default=None, metavar="DIR",
                    help="Checkpoint dir (if reachable from this host): assert exclude_state=true, use_relative_action=true "
                         "and the modality layout before moving the robot (also honours <dir>/processor/).")
    ap.add_argument("--reference-dataset", default=None, metavar="DIR",
                    help="[--check] LeRobot v2.1 dataset (e.g. trocar_cotrain_rel): dump the first frame of each camera of "
                         "--reference-episode to /tmp/check_rel_eef_dataset_*.png to compare the wrist labelling.")
    ap.add_argument("--reference-episode", type=int, default=0)
    ap.add_argument("--log-dir", default=None, help="Dump every chunk (state, server rows, IK trajectory, residuals) as .npz here.")

    # smoothing (same semantics as eval_h2_groot.py, applied to the IK joint targets)
    ap.add_argument("--smooth-alpha", type=float, default=1.0)
    ap.add_argument("--async-policy", action="store_true")
    ap.add_argument("--async-cache-poll-s", type=float, default=0.005)
    ap.add_argument("--boundary-smooth-alpha", type=float, default=1.0)
    ap.add_argument("--boundary-smooth-steps", type=int, default=0)
    ap.add_argument("--boundary-blend-steps", type=int, default=0)
    ap.add_argument("--boundary-blend-alpha", type=float, default=2.0)
    return ap.parse_args()


if __name__ == "__main__":
    eval_h2_groot_rel_eef(parse_args())
