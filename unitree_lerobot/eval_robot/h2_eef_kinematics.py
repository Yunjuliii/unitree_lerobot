"""
H2 + Sharpa end-effector kinematics for GR00T *relative end-effector* inference.

Frame / format conventions (identical to the training set ``trocar_cotrain_rel``,
built by isaac-ego-inpaint/scripts/make_trocar_cotrain_rel.py):

  * EEF          = URDF link frame ``{side}_hand_C_MC`` (Sharpa palm base), from
                   ``h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf``.
  * Reference    = the training data expressed the pose in the RIGHT-EYE head
                   camera optical frame ``head_camera_right_optical_frame`` (CV
                   optics: +x right, +y down, +z forward; its pelvis pose is fixed
                   by head_pitch 0.7 / head_yaw 0.095 rad, see
                   meta/head_camera_calibration.json of trocar_test_lerobot).
                   At inference the frame does NOT matter (next bullet), so
                   ``eef_frame="pelvis"`` (default) skips the camera transform:
                   FK, the state sent to the server, the server's absolute rows
                   and the IK targets are all pelvis-frame poses.
                   ``eef_frame="camera"`` reproduces the training-time state
                   values exactly (only relevant if the model were ever to see
                   the state, i.e. exclude_state not set on the checkpoint).
  * 9-dim layout = [x, y, z, R00, R01, R02, R10, R11, R12]
                   (GR00T ``ActionFormat.XYZ_ROT6D``: xyz + the first two ROWS
                   of the rotation matrix).
  * Relative     = GR00T ``ActionRepresentation.RELATIVE`` for ``ActionType.EEF``:
                   rel[k] = T_state^-1 @ T_action[k]; inverse: T_abs = T_state @ T_rel.
                   With both poses in one frame F rigidly attached to the pelvis,
                   T_F_hand(t)^-1 T_F_hand(t+k) = T_pelvis_hand(t)^-1 T_pelvis_hand(t+k):
                   the relative pose is the hand motion in the hand's own frame and
                   F cancels exactly (verified to 4e-16 on the training data), so the
                   model's outputs and the relative_stats normalisation are identical
                   for the camera and the pelvis frame.

Robot joint order (H2_JointArmIndex, what H2_ArmController reads / commands):
    [L shoulder_pitch, L shoulder_roll, L shoulder_yaw, L elbow,
     L wrist_roll, L wrist_pitch, L wrist_yaw,  R ... same 7]

Only numpy + pinocchio (>= 3.x) are needed; works in the ``tv`` conda env.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

SIDES = ("left", "right")
ARM_DOF = 14
EEF_DIM = 9

ARM_JOINT_NAMES = {
    side: [
        f"{side}_shoulder_pitch_joint",
        f"{side}_shoulder_roll_joint",
        f"{side}_shoulder_yaw_joint",
        f"{side}_elbow_joint",
        f"{side}_wrist_roll_joint",
        f"{side}_wrist_pitch_joint",
        f"{side}_wrist_yaw_joint",
    ]
    for side in SIDES
}
# robot 14-vector = left 7 then right 7 (H2_JointArmIndex order)
ROBOT_ARM_JOINT_NAMES = ARM_JOINT_NAMES["left"] + ARM_JOINT_NAMES["right"]
ROBOT_ARM_SLICE = {"left": slice(0, 7), "right": slice(7, 14)}

HAND_FRAME = {side: f"{side}_hand_C_MC" for side in SIDES}
CAMERA_FRAME = {
    "left": "head_camera_optical_frame",
    "right": "head_camera_right_optical_frame",
}
HEAD_PITCH_JOINT = "head_pitch_joint"
HEAD_YAW_JOINT = "head_yaw_joint"

# Head pose the training data was recorded / rendered with.
DEFAULT_HEAD_PITCH_RAD = 0.7
DEFAULT_HEAD_YAW_RAD = 0.095

_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_URDF_CANDIDATES = (
    os.path.join(_REPO_PARENT, "xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"),  # checkout next to unitree_lerobot
    "~/workspace/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf",
    "~/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf",
    "~/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf",
    "~/isaac-ego-inpaint/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf",
    "~/Fanxing/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf",
)


def resolve_urdf(path: str | None = None) -> str:
    """Return the H2+Sharpa URDF path: explicit arg > $H2_IK_URDF > $XR_TELEOPERATE_ROOT > defaults."""
    candidates: list[str] = []
    if path:
        candidates.append(path)
    if os.environ.get("H2_IK_URDF"):
        candidates.append(os.environ["H2_IK_URDF"])
    xr_root = os.environ.get("XR_TELEOPERATE_ROOT")
    if xr_root:
        candidates.append(os.path.join(xr_root, "h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"))
    candidates.extend(_URDF_CANDIDATES)
    for cand in candidates:
        full = os.path.abspath(os.path.expanduser(cand))
        if os.path.isfile(full):
            return full
    raise FileNotFoundError(
        "H2_with_sharpa_hands.urdf not found. Set H2_IK_URDF=/path/to/h2_with_sharpa/urdf/"
        f"H2_with_sharpa_hands.urdf (tried: {[os.path.expanduser(c) for c in candidates]})"
    )


# ── xyz + rot6d <-> homogeneous ───────────────────────────────────────────────

def xyz_rot6d_to_matrix(row9) -> np.ndarray:
    """(9,) [xyz, R row0, R row1] -> (4, 4).  Gram-Schmidt like GR00T's EndEffectorPose."""
    v = np.asarray(row9, dtype=np.float64).reshape(EEF_DIM)
    r0 = v[3:6]
    r1 = v[6:9]
    n0 = np.linalg.norm(r0)
    if n0 < 1e-8:
        raise ValueError("rot6d first row is (near) zero")
    r0 = r0 / n0
    r1 = r1 - np.dot(r0, r1) * r0
    n1 = np.linalg.norm(r1)
    if n1 < 1e-8:
        raise ValueError("rot6d rows are (near) parallel")
    r1 = r1 / n1
    r2 = np.cross(r0, r1)
    T = np.eye(4, dtype=np.float64)
    T[0, :3], T[1, :3], T[2, :3] = r0, r1, r2
    T[:3, 3] = v[:3]
    return T


def matrix_to_xyz_rot6d(T) -> np.ndarray:
    """(4, 4) -> (9,) float64 [xyz, R row0, R row1]."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([T[:3, 3], T[0, :3], T[1, :3]])


def rows_to_matrices(rows) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, EEF_DIM)
    return np.stack([xyz_rot6d_to_matrix(r) for r in rows], axis=0)


def compose_relative(T_ref, T_rel) -> np.ndarray:
    """GR00T relative EEF inverse: T_abs = T_ref @ T_rel (same frame as T_ref)."""
    return np.asarray(T_ref, dtype=np.float64) @ np.asarray(T_rel, dtype=np.float64)


def relative_chunk_to_absolute(ref_row9, rel_rows) -> np.ndarray:
    """(9,) reference state + (T, 9) relative chunk -> (T, 9) absolute chunk."""
    T_ref = xyz_rot6d_to_matrix(ref_row9)
    return np.stack(
        [matrix_to_xyz_rot6d(compose_relative(T_ref, xyz_rot6d_to_matrix(r))) for r in
         np.asarray(rel_rows, dtype=np.float64).reshape(-1, EEF_DIM)],
        axis=0,
    )


def pose_error(T_a, T_b) -> tuple[float, float]:
    """(translation error m, rotation error rad) between two homogeneous poses."""
    A = np.asarray(T_a, dtype=np.float64)
    B = np.asarray(T_b, dtype=np.float64)
    d_pos = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    d_rot = float(np.linalg.norm(pin.log3(A[:3, :3].T @ B[:3, :3])))
    return d_pos, d_rot


# ── IK result containers ──────────────────────────────────────────────────────

@dataclass
class IKStep:
    q: np.ndarray            # (7,)
    pos_err_m: float
    rot_err_rad: float
    iterations: int
    joint_step_rad: float    # max |q - q_init|
    clamped: bool = False    # joint-step guard engaged


@dataclass
class ChunkIK:
    q_traj: np.ndarray                       # (T, 14) robot order
    pos_err_m: np.ndarray                    # (T, 2)  left, right
    rot_err_rad: np.ndarray                  # (T, 2)
    iterations: np.ndarray                   # (T, 2)
    joint_step_rad: np.ndarray               # (T,)    max |dq| between consecutive steps
    clamped: np.ndarray                      # (T,) bool  joint-step guard engaged
    held: np.ndarray = None                  # (T, 2) bool  arm kept its previous joints (residual too large)
    solve_time_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.held is None:
            self.held = np.zeros(self.pos_err_m.shape, dtype=bool)

    def summary(self) -> str:
        return (
            f"IK {self.q_traj.shape[0]} steps in {self.solve_time_s * 1e3:.1f} ms | "
            f"pos err max L/R {self.pos_err_m[:, 0].max() * 1e3:.1f}/{self.pos_err_m[:, 1].max() * 1e3:.1f} mm, "
            f"rot err max L/R {np.degrees(self.rot_err_rad[:, 0].max()):.1f}/{np.degrees(self.rot_err_rad[:, 1].max()):.1f} deg, "
            f"max joint step {np.degrees(self.joint_step_rad.max()):.1f} deg, "
            f"clamped {int(self.clamped.sum())}, held {int(self.held.any(axis=1).sum())}"
        )


# ── kinematics ────────────────────────────────────────────────────────────────

class H2EefKinematics:
    """FK/IK for the two H2 arms with ``hand_C_MC`` targets in a head-camera frame."""

    def __init__(
        self,
        urdf_path: str | None = None,
        *,
        head_pitch_rad: float = DEFAULT_HEAD_PITCH_RAD,
        head_yaw_rad: float = DEFAULT_HEAD_YAW_RAD,
        eef_frame: str = "pelvis",
        camera_eye: str = "right",
        ik_damping: float = 1e-6,
        ik_max_iterations: int = 40,
        ik_tol_pos_m: float = 5e-4,
        ik_tol_rot_rad: float = np.deg2rad(0.2),
        ik_posture_weight: float = 0.0,
    ) -> None:
        if camera_eye not in CAMERA_FRAME:
            raise ValueError(f"camera_eye must be one of {tuple(CAMERA_FRAME)}, got {camera_eye!r}")
        if eef_frame not in ("pelvis", "camera"):
            raise ValueError(f"eef_frame must be 'pelvis' or 'camera', got {eef_frame!r}")
        self.eef_frame = eef_frame
        self.urdf_path = resolve_urdf(urdf_path)
        self.head_pitch_rad = float(head_pitch_rad)
        self.head_yaw_rad = float(head_yaw_rad)
        self.camera_eye = camera_eye
        self.camera_frame = CAMERA_FRAME[camera_eye]
        self.ik_damping = float(ik_damping)
        self.ik_max_iterations = int(ik_max_iterations)
        self.ik_tol_pos_m = float(ik_tol_pos_m)
        self.ik_tol_rot_rad = float(ik_tol_rot_rad)
        self.ik_posture_weight = float(ik_posture_weight)

        self.full_model = pin.buildModelFromUrdf(self.urdf_path)
        self.full_data = self.full_model.createData()
        for name in ROBOT_ARM_JOINT_NAMES + [HEAD_PITCH_JOINT, HEAD_YAW_JOINT]:
            if not self.full_model.existJointName(name):
                raise KeyError(f"URDF {self.urdf_path} has no joint {name!r}")
        for frame in list(HAND_FRAME.values()) + [self.camera_frame]:
            if not self.full_model.existFrame(frame):
                raise KeyError(
                    f"URDF {self.urdf_path} has no frame {frame!r}; the hand_C_MC / head camera "
                    "frames are only authored in h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"
                )

        # Reference configuration: every joint at zero (fixed base -- the URDF's floating joint is
        # commented out, so pelvis == world; fingers 0), head at the recording pose.  The arm
        # joints are the only free DoFs of the reduced models.
        self.q_ref = pin.neutral(self.full_model)
        self._set_joint(self.q_ref, HEAD_PITCH_JOINT, self.head_pitch_rad)
        self._set_joint(self.q_ref, HEAD_YAW_JOINT, self.head_yaw_rad)
        pin.forwardKinematics(self.full_model, self.full_data, self.q_ref)
        pin.updateFramePlacements(self.full_model, self.full_data)
        self.T_pelvis_camera = np.array(
            self.full_data.oMf[self.full_model.getFrameId(self.camera_frame)].homogeneous, dtype=np.float64
        )
        # Frame the 9-dim eef rows live in (state sent to the server, rows received from it).
        self.T_pelvis_ref = self.T_pelvis_camera.copy() if eef_frame == "camera" else np.eye(4, dtype=np.float64)
        self.T_ref_pelvis = np.linalg.inv(self.T_pelvis_ref)

        # One 7-DoF reduced model per arm (independent chains; torso/head fixed).
        # _arm[side] holds the Data used by the IK (main thread); _fk_data[side] is a separate Data
        # for the FK of the observation (async worker thread) -- pin.Model is read-only, so the two
        # threads never touch the same mutable buffers.
        self._arm: dict[str, tuple[pin.Model, pin.Data, int]] = {}
        self._fk_data: dict[str, pin.Data] = {}
        for side in SIDES:
            model = self._reduced(keep=set(ARM_JOINT_NAMES[side]))
            names = list(model.names[1:])
            if names != ARM_JOINT_NAMES[side]:
                raise RuntimeError(f"{side} arm reduced model joint order {names} != {ARM_JOINT_NAMES[side]}")
            if model.nq != 7 or model.nv != 7:
                raise RuntimeError(f"{side} arm reduced model has nq={model.nq}, expected 7")
            fid = model.getFrameId(HAND_FRAME[side])
            if fid >= model.nframes or model.frames[fid].name != HAND_FRAME[side]:
                raise KeyError(f"{HAND_FRAME[side]} missing from the reduced {side} arm model")
            self._arm[side] = (model, model.createData(), fid)
            self._fk_data[side] = model.createData()

        # 14-DoF dual-arm model for gravity feed-forward (includes the Sharpa hand masses).
        dual = self._reduced(keep=set(ROBOT_ARM_JOINT_NAMES))
        self._dual_model, self._dual_data = dual, dual.createData()
        # reduced-model q index of each robot-order joint
        self._dual_idx = np.array(
            [dual.idx_qs[dual.getJointId(n)] for n in ROBOT_ARM_JOINT_NAMES], dtype=np.int64
        )
        lo, hi = {}, {}
        for side in SIDES:
            model = self._arm[side][0]
            lo[side], hi[side] = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
        self.lower_limits = np.concatenate([lo["left"], lo["right"]])
        self.upper_limits = np.concatenate([hi["left"], hi["right"]])

    # -- helpers ----------------------------------------------------------------

    def _set_joint(self, q: np.ndarray, name: str, value: float) -> None:
        jid = self.full_model.getJointId(name)
        q[self.full_model.idx_qs[jid]] = float(value)

    def _reduced(self, keep: set[str]) -> pin.Model:
        locked = [
            jid for jid in range(1, self.full_model.njoints)
            if self.full_model.names[jid] not in keep
        ]
        return pin.buildReducedModel(self.full_model, locked, self.q_ref)

    @staticmethod
    def _split(q14) -> dict[str, np.ndarray]:
        q = np.asarray(q14, dtype=np.float64).reshape(ARM_DOF)
        return {side: q[ROBOT_ARM_SLICE[side]].copy() for side in SIDES}

    def describe(self) -> str:
        cam = self.T_pelvis_camera
        if self.eef_frame == "pelvis":
            ref = "PELVIS (camera transform skipped; relative poses are frame-invariant)"
        else:
            ref = (f"CAMERA {self.camera_frame} at head pitch/yaw {self.head_pitch_rad:.3f} / "
                   f"{self.head_yaw_rad:.3f} rad, T_pelvis_camera translation {cam[:3, 3].round(4).tolist()}")
        return (
            f"URDF={self.urdf_path}\n"
            f"  EEF frames: {HAND_FRAME['left']} / {HAND_FRAME['right']}\n"
            f"  eef reference frame: {ref}"
        )

    # -- forward kinematics -----------------------------------------------------

    def fk_pelvis(self, side: str, q7) -> np.ndarray:
        """T_pelvis_hand_C_MC (4, 4) for one arm (own Data: safe to call from the observation thread)."""
        model, _, fid = self._arm[side]
        data = self._fk_data[side]
        q = np.asarray(q7, dtype=np.float64).reshape(7)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return np.array(data.oMf[fid].homogeneous, dtype=np.float64)

    def fk_ref_matrices(self, q14) -> dict[str, np.ndarray]:
        """{side: T_ref_hand_C_MC (4, 4)} in the eef reference frame, from the 14 robot-order joints."""
        parts = self._split(q14)
        return {side: self.T_ref_pelvis @ self.fk_pelvis(side, parts[side]) for side in SIDES}

    def fk_ref(self, q14) -> tuple[np.ndarray, np.ndarray]:
        """(left9, right9) float32 -- the GR00T ``state.left_eef`` / ``state.right_eef`` values."""
        T = self.fk_ref_matrices(q14)
        return (
            matrix_to_xyz_rot6d(T["left"]).astype(np.float32),
            matrix_to_xyz_rot6d(T["right"]).astype(np.float32),
        )

    def ref_to_pelvis(self, T_ref_target) -> np.ndarray:
        """Reference-frame pose -> pelvis-frame pose (identity when eef_frame='pelvis')."""
        return self.T_pelvis_ref @ np.asarray(T_ref_target, dtype=np.float64)

    # -- inverse kinematics -----------------------------------------------------

    # LM tuning (radians / metres).  See _lm_solve.
    IK_LAMBDA_MIN = 1e-6
    IK_MAX_ITER_STEP_RAD = 0.25       # any joint, per LM iteration
    IK_MIN_GAIN = 1e-5                # stop instead of buying < 10 um / 10 urad with real joint motion ...
    IK_MIN_GAIN_PER_RAD = 2e-3        # ... or less than 2 mm (2 mrad) of error per radian of the largest joint move
    IK_RESTART_POS_ERR_M = 0.02       # only a clearly failed warm start gets reseeded ...
    IK_RESTART_ROT_ERR_RAD = np.deg2rad(10.0)
    IK_RESTART_DISTANCE_PENALTY = 20.0  # ... and a reseed must beat it by this many tol-units per max_joint_step of extra motion

    def _evaluate(self, side: str, q: np.ndarray, target: pin.SE3):
        """FK at q (IK Data); returns (T_current^-1 T_target, 6-vector log error)."""
        model, data, fid = self._arm[side]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        cur_from_target = data.oMf[fid].actInv(target)
        return cur_from_target, pin.log(cur_from_target).vector

    @staticmethod
    def _residual(cur_from_target: pin.SE3) -> tuple[float, float]:
        """(Euclidean translation error m, geodesic rotation error rad)."""
        return (
            float(np.linalg.norm(cur_from_target.translation)),
            float(np.linalg.norm(pin.log3(cur_from_target.rotation))),
        )

    def _lm_solve(self, side: str, target: pin.SE3, q_start: np.ndarray, posture, iters_max: int):
        """Levenberg-Marquardt on the SE(3) log error, robust near limits and the reach boundary.

        Returns (q, err6, iterations).
          * minimum-norm step: the 7-DoF arm's redundant coordinate stays where it
            was, so consecutive 30 Hz targets never flip IK branch;
          * joints sitting on a limit whose step points further out are removed
            from the Jacobian (active set) so the other joints take over instead of
            the clamp stalling the iteration;
          * a step is accepted only if the actual decrease is at least 10% of the
            decrease predicted by the linear model (LM gain ratio) -- otherwise the
            damping grows; every joint moves at most IK_MAX_ITER_STEP_RAD per
            iteration;
          * a step that buys less than IK_MIN_GAIN of error, or less than
            IK_MIN_GAIN_PER_RAD per radian of its largest joint move, stops the
            solve: this is what happens a few mm past the reach boundary, where the
            near-singular Jacobian would otherwise trade large self-motion for
            micrometres, and consecutive warm starts would wind the arm up along
            that direction (a well-conditioned arm gives ~0.3 m of error per rad).
        """
        model, data, fid = self._arm[side]
        lo, hi = model.lowerPositionLimit, model.upperPositionLimit
        eye6 = np.eye(6)
        q = q_start.copy()
        lam = max(self.ik_damping, self.IK_LAMBDA_MIN)
        cur_from_target, err = self._evaluate(side, q, target)
        it = 0
        while it < iters_max:
            if np.linalg.norm(err[:3]) < self.ik_tol_pos_m and np.linalg.norm(err[3:]) < self.ik_tol_rot_rad:
                break
            it += 1
            J_full = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)
            J_full = -pin.Jlog6(cur_from_target.inverse()) @ J_full
            e_norm = float(np.linalg.norm(err))
            accepted = False
            stop = False
            for _ in range(8):
                J = J_full
                for _pass in range(2):  # second pass after masking joints that push into a limit
                    inv = J.T @ np.linalg.solve(J @ J.T + lam * eye6, eye6)
                    step = -inv @ err
                    if posture is not None and self.ik_posture_weight > 0.0:
                        step = step + (np.eye(model.nv) - inv @ J) @ (self.ik_posture_weight * (posture - q))
                    blocked = ((q <= lo + 1e-6) & (step < 0.0)) | ((q >= hi - 1e-6) & (step > 0.0))
                    if not blocked.any() or _pass == 1:
                        step[blocked] = 0.0
                        break
                    J = J_full.copy()
                    J[:, blocked] = 0.0
                worst = float(np.max(np.abs(step)))
                if worst > self.IK_MAX_ITER_STEP_RAD:
                    step *= self.IK_MAX_ITER_STEP_RAD / worst
                q_new = np.clip(pin.integrate(model, q, step), lo, hi)
                dq = q_new - q
                cft_new, err_new = self._evaluate(side, q_new, target)
                actual = e_norm - float(np.linalg.norm(err_new))
                predicted = e_norm - float(np.linalg.norm(err + J_full @ dq))
                if actual > 0.0 and (predicted <= 0.0 or actual >= 0.1 * predicted):
                    dq_max = float(np.max(np.abs(dq)))
                    if dq_max > 1e-3 and (actual < self.IK_MIN_GAIN or actual < self.IK_MIN_GAIN_PER_RAD * dq_max):
                        stop = True  # negligible gain for real motion: (nearly) unreachable, do not wind up
                        break
                    q, cur_from_target, err = q_new, cft_new, err_new
                    lam = max(lam / 3.0, self.IK_LAMBDA_MIN)
                    accepted = True
                    break
                lam *= 10.0
            if stop or not accepted:
                self._evaluate(side, q, target)  # leave the FK cache at q
                break
        return q, err, it

    def solve_ik(
        self,
        side: str,
        T_pelvis_target,
        q_init,
        *,
        q_posture=None,
        max_iterations: int | None = None,
        max_joint_step_rad: float | None = None,
        restarts: int = 1,
    ) -> IKStep:
        """IK for one arm, warm-started from ``q_init``.

        Levenberg-Marquardt (see ``_lm_solve``).  Only if the warm start clearly
        failed (residual above IK_RESTART_POS_ERR_M / IK_RESTART_ROT_ERR_RAD, i.e.
        not a near-boundary target) up to ``restarts`` seeds around ``q_init`` are
        tried; a reseeded solution must beat the warm start by
        IK_RESTART_DISTANCE_PENALTY tolerance-units per ``max_joint_step_rad`` of
        extra joint motion, so a far IK branch cannot win on residual alone.  If
        ``max_joint_step_rad`` is given, a solution further than that from
        ``q_init`` (in any joint) is pulled back onto that ball and flagged
        ``clamped`` -- the caller's guard against branch flips within one 33 ms
        control step.  ``pos_err_m`` / ``rot_err_rad`` are the Euclidean /
        geodesic residuals of the returned joints.
        """
        model, _, _ = self._arm[side]
        target = pin.SE3(np.asarray(T_pelvis_target, dtype=np.float64))
        q0 = np.clip(np.asarray(q_init, dtype=np.float64).reshape(7), model.lowerPositionLimit, model.upperPositionLimit)
        posture = None if q_posture is None else np.asarray(q_posture, dtype=np.float64).reshape(7)
        iters_max = self.ik_max_iterations if max_iterations is None else int(max_iterations)

        q, _, it = self._lm_solve(side, target, q0, posture, iters_max)
        cft, _ = self._evaluate(side, q, target)
        pos, rot = self._residual(cft)
        if restarts > 0 and (pos > self.IK_RESTART_POS_ERR_M or rot > self.IK_RESTART_ROT_ERR_RAD):
            step_ref = max_joint_step_rad if (max_joint_step_rad is not None and max_joint_step_rad > 0) else 0.35

            def score(q_c, pos_c, rot_c):
                return (pos_c / self.ik_tol_pos_m + rot_c / self.ik_tol_rot_rad
                        + self.IK_RESTART_DISTANCE_PENALTY * float(np.max(np.abs(q_c - q0))) / step_ref)

            best = (score(q, pos, rot), q, pos, rot)
            rng = np.random.default_rng(int(abs(float(np.sum(q0)) * 1e4)) % (2**32))
            for _ in range(int(restarts)):
                seed = np.clip(q0 + rng.normal(0.0, 0.3, 7), model.lowerPositionLimit, model.upperPositionLimit)
                q_r, _, it_r = self._lm_solve(side, target, seed, posture, max(10, iters_max // 2))
                it += it_r
                cft_r, _ = self._evaluate(side, q_r, target)
                pos_r, rot_r = self._residual(cft_r)
                s = score(q_r, pos_r, rot_r)
                if s < best[0]:
                    best = (s, q_r, pos_r, rot_r)
            _, q, pos, rot = best

        clamped = False
        if max_joint_step_rad is not None and max_joint_step_rad > 0:
            dq = q - q0
            worst = float(np.max(np.abs(dq)))
            if worst > max_joint_step_rad:
                q = q0 + dq * (max_joint_step_rad / worst)
                clamped = True
        cft, _ = self._evaluate(side, q, target)
        pos, rot = self._residual(cft)
        return IKStep(
            q=q, pos_err_m=pos, rot_err_rad=rot, iterations=int(it),
            joint_step_rad=float(np.max(np.abs(q - q0))), clamped=clamped,
        )

    def solve_chunk(
        self,
        T_pelvis_left,
        T_pelvis_right,
        q14_init,
        *,
        q14_posture=None,
        max_joint_step_rad: float | None = None,
        hold_pos_err_m: float | None = None,
        hold_rot_err_rad: float | None = None,
    ) -> ChunkIK:
        """Sequential IK over a chunk of absolute PELVIS-frame targets.

        ``T_pelvis_left/right``: (T, 4, 4), or ``None`` for an arm that is not driven
        by the policy: that arm keeps its ``q14_init`` joints on every step (zero
        residual, zero motion), e.g. the left arm of a right-hand-only model.  Step 0
        is warm-started from ``q14_init`` (the arm's current / last commanded joints),
        every later step from the previous solution, which keeps the trajectory on one
        IK branch.

        ``hold_pos_err_m`` / ``hold_rot_err_rad``: if given, an arm whose residual
        exceeds them on step t keeps its previous joints (``held[t, arm]``) and the
        next step is warm-started from those held joints, so the per-step
        ``max_joint_step_rad`` guard is measured against what will actually be
        executed.  The reported residual of a held step is that of the held pose.
        """
        targets: dict[str, np.ndarray] = {}
        for side, T_side in (("left", T_pelvis_left), ("right", T_pelvis_right)):
            if T_side is not None:
                targets[side] = np.asarray(T_side, dtype=np.float64).reshape(-1, 4, 4)
        if not targets:
            raise ValueError("solve_chunk needs targets for at least one arm")
        lengths = {side: T.shape[0] for side, T in targets.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"left/right chunk lengths differ: {lengths}")
        n = next(iter(lengths.values()))
        q_prev = self._split(q14_init)
        posture = None if q14_posture is None else self._split(q14_posture)
        q_traj = np.zeros((n, ARM_DOF), dtype=np.float64)
        pos_err = np.zeros((n, 2))
        rot_err = np.zeros((n, 2))
        iters = np.zeros((n, 2), dtype=np.int64)
        joint_step = np.zeros(n)
        clamped = np.zeros(n, dtype=bool)
        held = np.zeros((n, 2), dtype=bool)
        failed_prev = {side: False for side in SIDES}  # no reseeding while an arm keeps missing (unreachable): bounded time
        t0 = time.perf_counter()
        for t in range(n):
            for j, side in enumerate(SIDES):
                if side not in targets:
                    q_traj[t, ROBOT_ARM_SLICE[side]] = q_prev[side]  # held arm: seed joints on every step
                    continue
                T_side = targets[side][t]
                step = self.solve_ik(
                    side, T_side, q_prev[side],
                    q_posture=None if posture is None else posture[side],
                    max_joint_step_rad=max_joint_step_rad,
                    restarts=0 if failed_prev[side] else 1,
                )
                failed_prev[side] = (step.pos_err_m > self.IK_RESTART_POS_ERR_M
                                     or step.rot_err_rad > self.IK_RESTART_ROT_ERR_RAD)
                bad = ((hold_pos_err_m is not None and step.pos_err_m > hold_pos_err_m)
                       or (hold_rot_err_rad is not None and step.rot_err_rad > hold_rot_err_rad))
                if bad:
                    q_hold = q_prev[side].copy()
                    cft, _ = self._evaluate(side, q_hold, pin.SE3(np.asarray(T_side, dtype=np.float64)))
                    p_h, r_h = self._residual(cft)
                    step = IKStep(q=q_hold, pos_err_m=p_h, rot_err_rad=r_h, iterations=step.iterations,
                                  joint_step_rad=0.0, clamped=False)
                    held[t, j] = True
                q_prev[side] = step.q
                q_traj[t, ROBOT_ARM_SLICE[side]] = step.q
                pos_err[t, j], rot_err[t, j], iters[t, j] = step.pos_err_m, step.rot_err_rad, step.iterations
                joint_step[t] = max(joint_step[t], step.joint_step_rad)
                clamped[t] |= step.clamped
        return ChunkIK(
            q_traj=q_traj, pos_err_m=pos_err, rot_err_rad=rot_err, iterations=iters,
            joint_step_rad=joint_step, clamped=clamped, held=held, solve_time_s=time.perf_counter() - t0,
        )

    def solve_ref_chunk(self, left_rows9, right_rows9, q14_init, **kwargs) -> ChunkIK:
        """Absolute reference-frame 9-dim rows (T, 9) per side -> joint trajectory.
        ``None`` for a side = that arm is held at its ``q14_init`` joints (see solve_chunk)."""
        TL = None if left_rows9 is None else np.stack([self.ref_to_pelvis(T) for T in rows_to_matrices(left_rows9)])
        TR = None if right_rows9 is None else np.stack([self.ref_to_pelvis(T) for T in rows_to_matrices(right_rows9)])
        return self.solve_chunk(TL, TR, q14_init, **kwargs)

    # -- dynamics ---------------------------------------------------------------

    def gravity_torque(self, q14) -> np.ndarray:
        """tau_ff = G(q) (N m) for the 14 arm joints, robot order, hands included."""
        q = np.asarray(q14, dtype=np.float64).reshape(ARM_DOF)
        q_model = np.zeros(self._dual_model.nq)
        q_model[self._dual_idx] = q
        tau = pin.rnea(
            self._dual_model, self._dual_data, q_model,
            np.zeros(self._dual_model.nv), np.zeros(self._dual_model.nv),
        )
        return np.asarray(tau, dtype=np.float64)[self._dual_idx]

    # API-compatible with eval_h2_groot._arm_tauff (H2_ArmIK.compute_tauff)
    def compute_tauff(self, q14, dq=None) -> np.ndarray:  # noqa: D401 - compatibility shim
        return self.gravity_torque(q14)


# ── server-output mode detection ──────────────────────────────────────────────

def detect_server_action_space(first_rows9, ref_row9, *, rel_pos_max_m: float = 0.08) -> tuple[str, str]:
    """Decide whether the server returned ABSOLUTE or RELATIVE eef poses.

    GR00T's processor composes relative->absolute on the server when
    ``use_relative_action`` is set (upstream launch_finetune always sets it).
    Absolute step-0 poses lie next to the reference state pose (the hands are
    0.3-0.6 m from the pelvis origin or the camera); relative ones lie next to
    the identity (|p| ~ mm, rot6d ~ [1,0,0,0,1,0]).
    Returns ("absolute"|"relative"|"ambiguous", reason).
    """
    rows = np.asarray(first_rows9, dtype=np.float64).reshape(-1, EEF_DIM)
    ref = np.asarray(ref_row9, dtype=np.float64).reshape(EEF_DIM)
    p0 = rows[0, :3]
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    d_abs = float(np.linalg.norm(p0 - ref[:3]))
    d_rel = float(np.linalg.norm(p0))
    rot_to_identity = float(np.linalg.norm(rows[0, 3:] - identity_rot6d))
    rot_to_ref = float(np.linalg.norm(rows[0, 3:] - ref[3:]))
    reason = (
        f"step0 |p|={d_rel:.3f} m, |p-p_state|={d_abs:.3f} m, "
        f"|rot6d-I|={rot_to_identity:.3f}, |rot6d-rot6d_state|={rot_to_ref:.3f}"
    )
    looks_relative = d_rel < rel_pos_max_m and d_rel < 0.5 * d_abs and rot_to_identity < rot_to_ref
    looks_absolute = d_abs < rel_pos_max_m and d_abs < 0.5 * d_rel and rot_to_ref < rot_to_identity
    if looks_relative and not looks_absolute:
        return "relative", reason
    if looks_absolute and not looks_relative:
        return "absolute", reason
    return "ambiguous", reason
