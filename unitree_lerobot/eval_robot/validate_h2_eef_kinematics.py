#!/usr/bin/env python3
"""
Offline validation of the relative-EEF inference math against recorded data.

No robot / server needed.  Uses the teleop dataset ``trocar_test_lerobot``
(LeRobot v3, one parquet), which stores both the 14 arm joints
(observation.state[0:14]) and the hand_C_MC poses in the right-eye camera
frame (observation.eef_pose_cam_right, computed from the same URDF at head
pitch 0.7 / yaw 0.095), and the relative-EEF training set
``trocar_cotrain_rel`` (state/action rows in the exact GR00T layout).

Checks
  1. FK:      camera-frame FK of the recorded q  ==  recorded eef_pose_cam_right  (frame + rot6d convention)
  2. IK:      sequential IK on one episode's recorded poses -> FK residual, joint deviation, speed
  3. Chunk:   simulate a GR00T chunk: state[t], action[t..t+31] from trocar_cotrain_rel
              -> GR00T's own relative_chunking / to_absolute_chunking (server side)
              -> client composition (relative_chunk_to_absolute) must agree
              -> IK -> joints vs the recorded joints of the same frames
  4. Detect:  detect_server_action_space tells absolute from relative chunks
  5. Frame:   the pelvis-frame pipeline (eef_frame='pelvis', the default at inference) gives
              the same relative chunk and the same IK joints as the camera-frame pipeline

Run in the tv env:
    python unitree_lerobot/eval_robot/validate_h2_eef_kinematics.py \
        --tele-dataset ~/isaac-ego-inpaint/dataset/trocar_test_lerobot \
        --rel-dataset  ~/isaac-ego-inpaint/dataset/trocar_cotrain_rel \
        --gr00t-root   ~/Isaac-GR00T
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h2_eef_kinematics import (  # noqa: E402
    ARM_DOF,
    DEFAULT_HEAD_PITCH_RAD,
    DEFAULT_HEAD_YAW_RAD,
    H2EefKinematics,
    detect_server_action_space,
    matrix_to_xyz_rot6d,
    pose_error,
    relative_chunk_to_absolute,
    rows_to_matrices,
)


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose7_to_matrix(p7):
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(p7[3:7])
    T[:3, 3] = p7[:3]
    return T


def col(table, name):
    return np.stack(table.column(name).to_numpy(zero_copy_only=False)).astype(np.float64)


def check_fk(kin_cam, tele_table, n_samples, rng):
    state = col(tele_table, "observation.state")
    pose14 = col(tele_table, "observation.eef_pose_cam_right")
    idx = rng.choice(len(state), size=min(n_samples, len(state)), replace=False)
    worst_pos, worst_rot = 0.0, 0.0
    for i in idx:
        T = kin_cam.fk_ref_matrices(state[i, :ARM_DOF])
        for side, start in (("left", 0), ("right", 7)):
            dp, dr = pose_error(T[side], pose7_to_matrix(pose14[i, start:start + 7]))
            worst_pos, worst_rot = max(worst_pos, dp), max(worst_rot, dr)
    ok = worst_pos < 1e-4 and worst_rot < 1e-3
    print(f"[1 FK] {len(idx)} rows: max pos err {worst_pos * 1e3:.4f} mm, max rot err {np.degrees(worst_rot):.4f} deg -> {'OK' if ok else 'FAIL'}")
    l9, r9 = kin_cam.fk_ref(state[idx[0], :ARM_DOF])
    print(f"       sample camera-frame state.left_eef  = {np.round(l9, 4).tolist()}")
    print(f"       sample camera-frame state.right_eef = {np.round(r9, 4).tolist()}")
    return ok


def check_ik_episode(kin_cam, tele_table, episode, max_frames):
    epi = tele_table.column("episode_index").to_numpy()
    rows = np.where(epi == episode)[0][:max_frames]
    state = col(tele_table, "observation.state")[rows]
    pose14 = col(tele_table, "observation.eef_pose_cam_right")[rows]
    TL = np.stack([kin_cam.ref_to_pelvis(pose7_to_matrix(p[0:7])) for p in pose14])
    TR = np.stack([kin_cam.ref_to_pelvis(pose7_to_matrix(p[7:14])) for p in pose14])
    q0 = state[0, :ARM_DOF]
    res = kin_cam.solve_chunk(TL, TR, q0)
    dq = np.abs(res.q_traj - state[:, :ARM_DOF])
    per_step_ms = res.solve_time_s * 1e3 / len(rows)
    ok = res.pos_err_m.max() < 3e-3 and res.rot_err_rad.max() < np.deg2rad(2.0)
    print(f"[2 IK] episode {episode}, {len(rows)} frames: {res.summary()}")
    print(f"       {per_step_ms:.3f} ms per frame (both arms), mean iterations L/R "
          f"{res.iterations[:, 0].mean():.1f}/{res.iterations[:, 1].mean():.1f}")
    print(f"       joint deviation from the recorded trajectory: max {np.degrees(dq.max()):.2f} deg, "
          f"mean {np.degrees(dq.mean()):.2f} deg (7-DoF redundancy; FK residual is the real metric) -> {'OK' if ok else 'FAIL'}")
    return ok


def load_rel_episode(rel_root, episode):
    info = json.load(open(os.path.join(rel_root, "meta/info.json")))
    path = os.path.join(rel_root, info["data_path"].format(episode_chunk=episode // 1000, episode_index=episode))
    tbl = pq.read_table(path)
    return col(tbl, "observation.state"), col(tbl, "action")


def check_chunk(rel_root, gr00t_root, episode, t):
    state, action = load_rel_episode(rel_root, episode)
    n = len(state)
    if t + 32 > n:
        t = max(0, n - 33)
    horizon = 32
    ok_all = True

    # GR00T's own conversion (what the loader does at training time / the server at inference)
    sys.path.insert(0, gr00t_root)
    from gr00t.data.state_action.action_chunking import EndEffectorActionChunk  # type: ignore
    from gr00t.data.state_action.pose import EndEffectorPose  # type: ignore
    from gr00t.data.types import ActionFormat  # type: ignore

    for side, sl in (("left", slice(0, 9)), ("right", slice(9, 18))):
        ref = state[t, sl]
        abs_chunk = action[t:t + horizon, sl]                       # absolute on disk
        chunk = EndEffectorActionChunk.from_array(abs_chunk, ActionFormat.XYZ_ROT6D)
        ref_pose = EndEffectorPose.from_action_format(ref, ActionFormat.XYZ_ROT6D)
        rel = chunk.relative_chunking(reference_frame=ref_pose).to(ActionFormat.XYZ_ROT6D)   # model target
        back = EndEffectorActionChunk.from_array(rel, ActionFormat.XYZ_ROT6D).to_absolute_chunking(
            reference_frame=ref_pose).to(ActionFormat.XYZ_ROT6D)                                # server output
        mine = relative_chunk_to_absolute(ref, rel)                                              # client fallback
        e_server = np.abs(back - abs_chunk).max()
        e_client = np.abs(mine - abs_chunk).max()
        step0 = np.linalg.norm(rel[0, :3])
        print(f"[3 chunk] {side}: |T_state^-1 T_action[0]| = {step0 * 1e3:.2f} mm; "
              f"server round-trip err {e_server:.2e}; client composition err {e_client:.2e}")
        ok_all &= e_server < 1e-6 and e_client < 1e-6
        mode, _ = detect_server_action_space(back, ref)
        mode_rel, _ = detect_server_action_space(rel, ref)
        print(f"[4 detect] {side}: absolute chunk -> {mode!r}; relative chunk -> {mode_rel!r}")
        ok_all &= mode == "absolute" and mode_rel == "relative"
    return ok_all, t


def check_chunk_ik(kin_cam, tele_table, rel_root, episode, t):
    _, action = load_rel_episode(rel_root, episode)
    epi = tele_table.column("episode_index").to_numpy()
    rows = np.where(epi == episode)[0]
    q_rec = col(tele_table, "observation.state")[rows][:, :ARM_DOF]
    horizon = 32
    q0 = q_rec[t]
    res = kin_cam.solve_ref_chunk(action[t:t + horizon, 0:9], action[t:t + horizon, 9:18], q0, max_joint_step_rad=0.35)
    # action[t+k] = state[t+1+k] -> compare with recorded joints at t+1+k
    q_target = q_rec[t + 1:t + 1 + horizon]
    dq = np.abs(res.q_traj - q_target)
    ok = res.pos_err_m.max() < 3e-3 and res.rot_err_rad.max() < np.deg2rad(2.0)
    print(f"[3 chunk IK] episode {episode} t={t}: {res.summary()}")
    print(f"             joints vs recorded frames t+1..t+32: max {np.degrees(dq.max()):.2f} deg, mean {np.degrees(dq.mean()):.2f} deg -> {'OK' if ok else 'FAIL'}")
    return ok, res.q_traj


def check_pelvis_frame(kin_pel, kin_cam, tele_table, rel_root, episode, t, q_traj_cam):
    """The inference default: FK/state/IK all in the pelvis frame. Must match the camera-frame path."""
    state, action = load_rel_episode(rel_root, episode)
    epi = tele_table.column("episode_index").to_numpy()
    q_rec = col(tele_table, "observation.state")[epi == episode][:, :ARM_DOF]
    horizon = 32
    ok = True
    # (a) the state the two pipelines would send differ only by the fixed camera transform
    l_pel, r_pel = kin_pel.fk_ref(q_rec[t])
    l_cam, r_cam = kin_cam.fk_ref(q_rec[t])
    T_l = kin_cam.T_pelvis_camera @ rows_to_matrices(l_cam[None])[0]
    e_state = np.abs(matrix_to_xyz_rot6d(T_l) - l_pel).max()
    # (b) the relative chunk (what the model predicts) is identical in both frames
    for sl in (slice(0, 9), slice(9, 18)):
        T_cam = rows_to_matrices(np.vstack([state[t, sl], action[t:t + horizon, sl]]))
        T_pel = np.einsum("ij,njk->nik", kin_cam.T_pelvis_camera, T_cam)
        rel_cam = np.einsum("ij,njk->nik", np.linalg.inv(T_cam[0]), T_cam[1:])
        rel_pel = np.einsum("ij,njk->nik", np.linalg.inv(T_pel[0]), T_pel[1:])
        ok &= np.abs(rel_cam - rel_pel).max() < 1e-9
    e_rel = np.abs(rel_cam - rel_pel).max()
    # (c) server-side composition onto the pelvis-frame state + pelvis-frame IK == camera path joints
    abs_pel = {}
    for side, sl, ref_row in (("left", slice(0, 9), l_pel), ("right", slice(9, 18), r_pel)):
        T_cam = rows_to_matrices(np.vstack([state[t, sl], action[t:t + horizon, sl]]))
        rel = np.einsum("ij,njk->nik", np.linalg.inv(T_cam[0]), T_cam[1:])
        abs_pel[side] = relative_chunk_to_absolute(ref_row.astype(np.float64), np.stack([matrix_to_xyz_rot6d(R) for R in rel]))
    res = kin_pel.solve_ref_chunk(abs_pel["left"], abs_pel["right"], q_rec[t], max_joint_step_rad=0.35)
    e_q = np.degrees(np.abs(res.q_traj - q_traj_cam).max())
    ok &= e_state < 1e-5 and e_q < 1e-3 and res.pos_err_m.max() < 3e-3
    print(f"[5 pelvis frame] state differs from camera-frame state only by T_pelvis_camera (err {e_state:.1e}); "
          f"relative chunk identical in both frames (max |diff| {e_rel:.1e}); "
          f"pelvis-frame IK joints vs camera-frame IK joints: max {e_q:.2e} deg; {res.summary()} -> {'OK' if ok else 'FAIL'}")
    return ok


def write_init_posture_yaml(tele_table, out_path, first_n=3):
    """Average of the first ``first_n`` frames of every episode, in the YAML layout
    of teleop/utils/compute_average_initial_state.py (what --init-state-yaml reads)."""
    import yaml
    epi = tele_table.column("episode_index").to_numpy()
    fi = tele_table.column("frame_index").to_numpy()
    state = col(tele_table, "observation.state")
    mask = fi < first_n
    s = state[mask]
    payload = {
        "first_n_frames": int(first_n),
        "num_episodes": int(len(np.unique(epi))),
        "num_frames": int(mask.sum()),
        "source_dataset": "isaac-ego-inpaint/dataset/trocar_test_lerobot (observation.state, 58-dim)",
        "states": {
            "left_arm": {"qpos": s[:, 0:7].mean(0).tolist()},
            "right_arm": {"qpos": s[:, 7:14].mean(0).tolist()},
            "left_ee": {"qpos": s[:, 14:36].mean(0).tolist()},
            "right_ee": {"qpos": s[:, 36:58].mean(0).tolist()},
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[init] wrote {out_path}: {payload['num_episodes']} episodes x {first_n} frames; "
          f"left_arm {np.round(payload['states']['left_arm']['qpos'], 3).tolist()} "
          f"right_arm {np.round(payload['states']['right_arm']['qpos'], 3).tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tele-dataset", default=os.path.expanduser("~/isaac-ego-inpaint/dataset/trocar_test_lerobot"))
    ap.add_argument("--rel-dataset", default=os.path.expanduser("~/isaac-ego-inpaint/dataset/trocar_cotrain_rel"))
    ap.add_argument("--gr00t-root", default=os.path.expanduser("~/Isaac-GR00T"))
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--t", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--fk-samples", type=int, default=400)
    ap.add_argument("--write-init-yaml", default=None, help="Write the average initial posture YAML here.")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    kin_cam = H2EefKinematics(args.urdf, eef_frame="camera", head_pitch_rad=DEFAULT_HEAD_PITCH_RAD, head_yaw_rad=DEFAULT_HEAD_YAW_RAD)
    kin_pel = H2EefKinematics(args.urdf, eef_frame="pelvis")
    print(kin_cam.describe())
    print(kin_pel.describe())
    tele = pq.read_table(os.path.join(args.tele_dataset, "data/chunk-000/file-000.parquet"))
    ok = True
    ok &= check_fk(kin_cam, tele, args.fk_samples, rng)
    ok &= check_ik_episode(kin_cam, tele, args.episode, args.max_frames)
    if os.path.isdir(args.rel_dataset) and os.path.isdir(args.gr00t_root):
        ok_chunk, t = check_chunk(args.rel_dataset, args.gr00t_root, args.episode, args.t)
        ok &= ok_chunk
        ok_ik, q_traj_cam = check_chunk_ik(kin_cam, tele, args.rel_dataset, args.episode, t)
        ok &= ok_ik
        ok &= check_pelvis_frame(kin_pel, kin_cam, tele, args.rel_dataset, args.episode, t, q_traj_cam)
    else:
        print("[3/4/5] skipped (rel dataset or Isaac-GR00T not found)")
    if args.write_init_yaml:
        write_init_posture_yaml(tele, args.write_init_yaml)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
