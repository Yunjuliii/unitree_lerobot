#!/usr/bin/env python3
"""
Open-loop replay of a recorded episode through a running GR00T server, using the
SAME observation building / decoding / IK code path as eval_h2_groot_rel_eef.py --
no robot, no cameras.

For every query frame t (every --stride frames) it
  1. reads the three camera frames of the episode (head_right -> video.high, wrists),
  2. builds state.left_eef / right_eef by FK of the RECORDED arm joints (teleop
     episodes 0..100 of trocar_test_lerobot) or, for the ego episodes without joints,
     from the dataset's own eef rows moved into the chosen --eef-frame,
  3. sends the observation over ZMQ, decodes the chunk (absolute / relative detection,
     composition, IK) exactly like the eval script,
  4. compares the model's 32 predicted poses with the recorded future poses
     (position / rotation error per step), the hand targets with the recorded hand
     joints, and -- when joints are available -- the IK joints with the recorded joints.

Start the server first (this machine has the checkpoint + Isaac-GR00T venv):
    cd ~/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \\
        --model-path checkpoints/gr00t-n1.7-h2-sharpa-cotrain-rel-ckpt42000 \\
        --embodiment-tag new_embodiment --port 5555
then (tv env):
    python replay_dataset_rel_eef.py --episode 3 --stride 16
    python replay_dataset_rel_eef.py --episode 150 --stride 16          # an ego (inpainted) episode
Results are printed and saved to --out (npz) for plotting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from eval_h2_groot import GR00TClient, _add_bt_dims  # noqa: E402
from eval_h2_groot_rel_eef import (  # noqa: E402
    ACTION_HAND_KEYS,
    LANGUAGE_KEY,
    ObsRef,
    _eef_rows,
    absolute_ref_rows,
    report_bad_steps,
    resolve_server_mode,
)
from h2_eef_kinematics import (  # noqa: E402
    ARM_DOF,
    DEFAULT_HEAD_PITCH_RAD,
    DEFAULT_HEAD_YAW_RAD,
    H2EefKinematics,
    matrix_to_xyz_rot6d,
    pose_error,
    rows_to_matrices,
)

VIDEO_KEYS = {"high": "observation.images.head_right", "left_wrist_view": "observation.images.wrist_left",
              "right_wrist_view": "observation.images.wrist_right"}
N_TELE = 101  # trocar_cotrain_rel episodes 0..100 come from trocar_test_lerobot (joints available)


def read_frames(path: str, indices: list[int]) -> dict[int, np.ndarray]:
    import cv2
    want = set(indices)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    out, i = {}, 0
    while want:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = bgr[:, :, ::-1].copy()  # RGB like the live client
            want.discard(i)
        i += 1
    cap.release()
    if want:
        raise RuntimeError(f"{path}: frames {sorted(want)} beyond the video length ({i} frames)")
    return out


def load_episode(root: str, episode: int):
    info = json.load(open(os.path.join(root, "meta/info.json")))
    import pyarrow.parquet as pq
    tbl = pq.read_table(os.path.join(root, info["data_path"].format(episode_chunk=episode // 1000, episode_index=episode)))
    col = lambda n: np.stack(tbl.column(n).to_numpy(zero_copy_only=False)).astype(np.float64)  # noqa: E731
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(os.path.join(root, "meta/tasks.jsonl")) if l.strip()}
    task = tasks[int(tbl.column("task_index").to_numpy()[0])]
    videos = {k: os.path.join(root, info["video_path"].format(episode_chunk=episode // 1000, video_key=v, episode_index=episode))
              for k, v in VIDEO_KEYS.items()}
    return col("observation.state"), col("action"), task, videos


def load_joints(tele_root: str, episode: int) -> np.ndarray | None:
    path = os.path.join(tele_root, "data/chunk-000/file-000.parquet")
    if episode >= N_TELE or not os.path.isfile(path):
        return None
    import pyarrow.parquet as pq
    tbl = pq.read_table(path, columns=["episode_index", "observation.state"])
    epi = tbl.column("episode_index").to_numpy()
    q = np.stack(tbl.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float64)[epi == episode][:, :ARM_DOF]
    return q if len(q) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.expanduser("~/isaac-ego-inpaint/dataset/trocar_cotrain_rel"))
    ap.add_argument("--tele-dataset", default=os.path.expanduser("~/isaac-ego-inpaint/dataset/trocar_test_lerobot"))
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--stride", type=int, default=16, help="frames between two queries")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--policy-host", default="localhost")
    ap.add_argument("--policy-port", type=int, default=5555)
    ap.add_argument("--timeout-ms", type=int, default=30000)
    ap.add_argument("--eef-frame", choices=["pelvis", "camera"], default="pelvis")
    ap.add_argument("--server-returns", choices=["auto", "absolute", "relative"], default="auto")
    ap.add_argument("--task", default=None, help="override the episode's task text")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--out", default=None, help="npz with per-query predictions / errors (default: outputs next to this script)")
    args = ap.parse_args()

    state, action, task, videos = load_episode(args.dataset, args.episode)
    task = args.task or task
    n = len(state)
    joints = load_joints(args.tele_dataset, args.episode)
    kin = H2EefKinematics(args.urdf, eef_frame=args.eef_frame, head_pitch_rad=DEFAULT_HEAD_PITCH_RAD, head_yaw_rad=DEFAULT_HEAD_YAW_RAD)
    T_ref_cam = kin.T_ref_pelvis @ kin.T_pelvis_camera  # dataset rows (camera frame) -> chosen reference frame

    def cam_rows_to_ref(rows9):
        return np.stack([matrix_to_xyz_rot6d(T_ref_cam @ T) for T in rows_to_matrices(rows9)])

    horizon = 32
    queries = list(range(0, n - horizon - 1, max(1, args.stride)))
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    print(f"episode {args.episode}: {n} frames, task {task!r}, joints {'available (FK state)' if joints is not None else 'n/a (dataset eef rows as state)'}, "
          f"{len(queries)} queries, eef_frame={args.eef_frame}")
    frames = {k: read_frames(p, queries) for k, p in videos.items()}

    client = GR00TClient(host=args.policy_host, port=args.policy_port, timeout_ms=args.timeout_ms)
    if not client.ping():
        print(f"ERROR: no GR00T server at {args.policy_host}:{args.policy_port}", file=sys.stderr)
        return 1
    client.reset()

    server_mode = args.server_returns
    rows_pos, rows_rot, rows_hand, rows_ik, rows_dq, rows_t, rows_infer = [], [], [], [], [], [], []
    pred_all, gt_all = [], []
    for qi, t in enumerate(queries):
        if joints is not None:
            arm_q = joints[t]
            l9, r9 = kin.fk_ref(arm_q)
        else:
            arm_q = None
            l9 = cam_rows_to_ref(state[t:t + 1, 0:9])[0].astype(np.float32)
            r9 = cam_rows_to_ref(state[t:t + 1, 9:18])[0].astype(np.float32)
        obs = {
            "video": {k: frames[k][t] for k in VIDEO_KEYS},
            "state": {"left_eef": l9.astype(np.float32), "right_eef": r9.astype(np.float32),
                      "left_hand": state[t, 18:40].astype(np.float32), "right_hand": state[t, 40:62].astype(np.float32)},
            "language": {LANGUAGE_KEY: task},
        }
        ref = ObsRef(arm_q=arm_q if arm_q is not None else np.zeros(ARM_DOF), left_eef=l9.astype(np.float64),
                     right_eef=r9.astype(np.float64), t_obs=time.monotonic())
        t0 = time.perf_counter()
        chunk, _ = client.get_action(_add_bt_dims(obs))
        infer_ms = (time.perf_counter() - t0) * 1e3
        if server_mode == "auto":
            server_mode = resolve_server_mode(chunk, ref, "auto")
        abs_rows = absolute_ref_rows(chunk, ref, server_mode)
        T_pred = {s: rows_to_matrices(abs_rows[s]) for s in ("left", "right")}
        gt = {"left": cam_rows_to_ref(action[t:t + horizon, 0:9]), "right": cam_rows_to_ref(action[t:t + horizon, 9:18])}
        T_gt = {s: rows_to_matrices(gt[s]) for s in ("left", "right")}
        pos = np.zeros((horizon, 2)); rot = np.zeros((horizon, 2))
        for j, s in enumerate(("left", "right")):
            for k in range(horizon):
                pos[k, j], rot[k, j] = pose_error(T_pred[s][k], T_gt[s][k])
        hand = np.zeros((horizon, 2))
        for j, s in enumerate(("left", "right")):
            key = ACTION_HAND_KEYS[s]
            if key in chunk:
                pred_h = np.asarray(chunk[key][0][:horizon], dtype=np.float64)
                hand[:, j] = np.abs(pred_h - action[t:t + horizon, 18 + 22 * j:40 + 22 * j]).max(axis=1)
        line = (f"t={t:4d} infer {infer_ms:5.0f} ms | pos err mm L/R: step0 {pos[0,0]*1e3:5.1f}/{pos[0,1]*1e3:5.1f}, "
                f"step15 {pos[15,0]*1e3:5.1f}/{pos[15,1]*1e3:5.1f}, step31 {pos[31,0]*1e3:5.1f}/{pos[31,1]*1e3:5.1f} | "
                f"rot err deg step15 {np.degrees(rot[15,0]):4.1f}/{np.degrees(rot[15,1]):4.1f} | hand max |dq| deg step15 "
                f"{np.degrees(hand[15,0]):4.1f}/{np.degrees(hand[15,1]):4.1f}")
        if joints is not None:
            res = kin.solve_ref_chunk(abs_rows["left"], abs_rows["right"], arm_q, max_joint_step_rad=0.35)
            report_bad_steps(res, 0.03, np.deg2rad(20), False)
            dq = np.degrees(np.abs(res.q_traj - joints[t + 1:t + 1 + horizon]))
            line += f" | IK resid max {res.pos_err_m.max()*1e3:4.1f} mm, joints vs recorded max {dq.max():4.1f} deg (step15 {dq[15].max():4.1f})"
            rows_ik.append([res.pos_err_m.max(), res.rot_err_rad.max()]); rows_dq.append(dq.max(axis=1))
        print(line)
        rows_pos.append(pos); rows_rot.append(rot); rows_hand.append(hand); rows_t.append(t); rows_infer.append(infer_ms)
        pred_all.append(np.concatenate([abs_rows["left"], abs_rows["right"]], axis=1)); gt_all.append(np.concatenate([gt["left"], gt["right"]], axis=1))

    P = np.stack(rows_pos) * 1e3; R = np.degrees(np.stack(rows_rot)); H = np.degrees(np.stack(rows_hand))
    print("\n=== open-loop summary over all queries (model prediction vs recorded future) ===")
    for k in (0, 7, 15, 31):
        print(f"step {k:2d}: pos err mm mean L/R {P[:,k,0].mean():5.1f}/{P[:,k,1].mean():5.1f} (p90 {np.percentile(P[:,k,0],90):5.1f}/{np.percentile(P[:,k,1],90):5.1f}) | "
              f"rot err deg mean {R[:,k,0].mean():4.1f}/{R[:,k,1].mean():4.1f} | hand max|dq| deg mean {H[:,k,0].mean():4.1f}/{H[:,k,1].mean():4.1f}")
    if rows_ik:
        IK = np.array(rows_ik); DQ = np.stack(rows_dq)
        print(f"IK: residual max over queries {IK[:,0].max()*1e3:.1f} mm / {np.degrees(IK[:,1].max()):.1f} deg; "
              f"IK joints vs recorded: mean over steps {DQ.mean():.1f} deg, max {DQ.max():.1f} deg")
    print(f"inference {np.mean(rows_infer):.0f} ms mean, server mode {server_mode}")
    out = args.out or os.path.join(_EVAL_DIR, "outputs", f"replay_ep{args.episode:03d}_{args.eef_frame}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, t=np.array(rows_t), pos_err_m=np.stack(rows_pos), rot_err_rad=np.stack(rows_rot),
                        hand_err_rad=np.stack(rows_hand), pred_eef=np.stack(pred_all), gt_eef=np.stack(gt_all),
                        infer_ms=np.array(rows_infer), server_mode=server_mode, eef_frame=args.eef_frame, task=task,
                        ik_dq_deg=np.stack(rows_dq) if rows_dq else np.zeros(0))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
