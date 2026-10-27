# eval_h2_groot_rel_eef.py — H2 + Sharpa inference for the relative-EEF / no-state GR00T model

Drives the H2 robot with a GR00T checkpoint trained on `trocar_cotrain_rel`
(`isaac-ego-inpaint/cluster/train_gr00t_cotrain_rel.sbatch`): actions are
**relative end-effector poses** of the Sharpa base (URDF `hand_C_MC`), hands
are absolute 22-joint targets, and the model was trained with
`state_dropout 1.0` (it never sees the state). The training set expressed the
poses in the right-eye head camera frame; at inference the pelvis frame is
used by default because relative poses are frame-invariant (see below).

Files (all in this directory):

| File | Role |
|---|---|
| `run_eval_h2_groot_rel_eef.sh` | launcher, same shape as `run_eval_h2_groot.sh` |
| `eval_h2_groot_rel_eef.py` | the inference loop (imports camera / hand / ZMQ helpers from `eval_h2_groot.py`) |
| `h2_eef_kinematics.py` | pinocchio FK/IK on `h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf`: `hand_C_MC` poses in the pelvis (default) or right-eye camera frame, rot6d helpers, relative composition, gravity torque |
| `patch_groot_processor_exclude_state.py` | sets `exclude_state: true` in a checkpoint's `processor_config.json` (server side, run once) |
| `validate_h2_eef_kinematics.py` | offline check of FK / IK / relative composition against the recorded datasets (no robot) |
| `init_post_trocar.yaml` | mean of the first 3 frames of the 101 trocar teleop episodes (`--init-state-yaml` default) |

## Control flow per chunk

```
arm joints q (14, H2_JointArmIndex order)
   │ FK (pinocchio, h2_with_sharpa URDF) — PELVIS frame by default (--eef-frame pelvis)
   ▼
T_pelvis_hand_C_MC  →  [x,y,z, R00,R01,R02, R10,R11,R12]  = state.left_eef / state.right_eef
   │ + video.high (right eye), video.left_wrist_view, video.right_wrist_view, hands, language
   ▼
GR00T server: model predicts 32 RELATIVE poses  T_rel[k] = T_state^-1 T_action[k]   (k = 0..31, all anchored
              on the observation pose); the processor composes T_abs[k] = T_state @ T_rel[k] and returns
              ABSOLUTE poses in the frame of the state it was given (use_relative_action=True, set by launch_finetune).
   │
   ▼  --server-returns auto  (detects absolute vs raw relative from the first chunk; client composes if needed)
T_pelvis_target[k] = T_abs[k]            (with --eef-frame camera: T_pelvis_cam @ T_cam_abs[k])
   │ sequential Levenberg–Marquardt IK per arm, warm-started from the last commanded joints
   ▼
q_target[k] (14)  →  H2_ArmController.ctrl_dual_arm  at 30 Hz;  hands ← action.left_hand / right_hand
```

Timing: `action[t] = state[t+1]` in the dataset, so chunk step `k` is the pose
expected `(k+1)/30 s` after the observation. `--action-horizon 16` of the 32
steps are executed before re-querying (same as the joint-space script);
`--async-policy` keeps inferring in the background and skips the steps that
elapsed during inference.

**Why the pelvis frame is fine although training used the right-eye camera
frame.** For any frame F rigidly attached to the pelvis,
`T_F_hand(t)^-1 T_F_hand(t+k) = T_pelvis_hand(t)^-1 T_pelvis_hand(t+k)`: the
relative pose is the hand motion in the hand's own frame and F cancels
(checked on the training data: 4e-16). The model's outputs and the
`relative_stats` normalisation are therefore identical in both frames, and
the server composes onto whatever state frame it receives. The state is only
that composition anchor; the model must not see it (next section).
`--eef-frame camera` (with `--cam-head-pitch 0.7 --cam-head-yaw 0.095`, the
recording calibration) reproduces the training-time state values exactly and
only matters if the model could ever see the state.

## Before the first run

1. **Checkpoint: hide the state from the model.** GR00T applies
   `state_dropout_prob` only while training; at inference the processor feeds
   the real normalized state unless `exclude_state` is set. On the server host:

   ```bash
   python patch_groot_processor_exclude_state.py /path/to/ckpt_trocar_cotrain_rel_sd1.0_bs1024/checkpoint-XXXXX
   # prints exclude_state / use_relative_action / modality keys, keeps processor_config.json.orig
   ```

   If the checkpoint directory is reachable from the eval host, set
   `REL_EEF_CKPT=<dir>` for the launcher (or pass `--verify-checkpoint <dir>`):
   the eval script then aborts when `exclude_state` or `use_relative_action` is
   not true or the modality layout differs. Without it the script can only warn:
   the server exposes no signal about its processor config. Both scripts also
   accept the `<dir>/processor/` layout newer trainers write.

   `use_relative_action` must stay `true`: that one flag selects both the
   relative→absolute composition and the `relative_action` statistics the eef
   groups are un-normalised with. A processor without it returns garbage rows,
   which is why `--server-returns auto` aborts instead of composing client-side
   when the first chunk looks relative.

1b. **Wrist camera labelling.** `trocar_cotrain_rel`'s 101 teleop episodes carry
   the wrist labels of the recording session (the `*_wrist_swaped` datasets were
   made because those were once found mirrored). Check once which labelling the
   live robot matches and add `--swap-wrist-cams` if needed:

   ```bash
   python eval_h2_groot_rel_eef.py --check --task "..." --reference-dataset ~/isaac-ego-inpaint/dataset/trocar_cotrain_rel
   # compare /tmp/check_rel_eef_left_wrist_view.png with /tmp/check_rel_eef_dataset_left_wrist_view.png
   ```

2. **Server**

   ```bash
   cd ~/Isaac-GR00T
   uv run python gr00t/eval/run_gr00t_server.py --model-path <checkpoint dir> --embodiment-tag new_embodiment --port 5555
   ```

3. **Offline sanity check of the kinematics** (any host with the datasets, tv env):

   ```bash
   python validate_h2_eef_kinematics.py --tele-dataset ~/isaac-ego-inpaint/dataset/trocar_test_lerobot \
       --rel-dataset ~/isaac-ego-inpaint/dataset/trocar_cotrain_rel --gr00t-root ~/Isaac-GR00T
   ```

   Verified on 2026-09-14: camera-frame FK reproduces
   `observation.eef_pose_cam_right` of the teleop dataset exactly (0.0 mm /
   0.0°); GR00T's own relative→absolute round trip and the client-side
   composition agree to 4e-8; sequential IK over a 32-step chunk: ~0.15 ms per
   step for both arms, residual < 0.5 mm / 0.2°, joints within 1.5° of the
   recorded trajectory; the pelvis-frame pipeline gives the same relative chunk
   and the same IK joints as the camera-frame one.

4. **One observation without the server**

   ```bash
   python eval_h2_groot_rel_eef.py --check --task "pick up the trocar and place it in the tray" --head-eye right
   ```

   Prints the FK eef state, an IK round trip on the current pose and saves
   `/tmp/check_rel_eef_*.png`.

## Run

```bash
bash run_eval_h2_groot_rel_eef.sh "pick up the trocar and place it in the tray" 300
bash run_eval_h2_groot_rel_eef.sh "..." 300 --async-policy --log-dir ~/rel_eef_logs
```

Defaults set by the launcher: `--motion --head-eye right --head-pitch-home 0.7
--hands-source dds --image-source dds --camera-dds-domain 10 --hold-s 20
--action-horizon 16 --eef-frame pelvis --server-returns auto
--init-state-yaml init_post_trocar.yaml`.
Environment: `H2_IK_URDF` (URDF path), `INIT_STATE_YAML`.

## Flags specific to this script

| Flag | Default | Meaning |
|---|---|---|
| `--eef-frame {pelvis,camera}` | pelvis | frame of the 9-dim eef state / targets; `camera` = right-eye camera frame exactly as in training |
| `--server-returns {auto,absolute,relative}` | auto | what the server's `*_eef` rows are; `auto` checks the first chunk (absolute rows sit next to the state pose, relative rows next to the identity) and aborts on relative rows (misconfigured processor); `relative` composes client-side anyway, debugging only |
| `--reference-dataset DIR` / `--reference-episode N` | – | `--check` only: dump the first frame of each camera of a training episode next to the live snapshots (wrist labelling check) |
| `--cam-head-pitch / --cam-head-yaw` | 0.7 / 0.095 | `--eef-frame camera` only: head joints used for the camera-frame FK (the training calibration) |
| `--cam-head-from-robot` | off | `--eef-frame camera` only: read the head joints from `rt/lowstate` once instead |
| `--max-joint-step-rad` | 0.35 | IK guard: max joint change between consecutive chunk steps (branch-flip protection) |
| `--ik-max-pos-err-m / --ik-max-rot-err-deg` | 0.03 / 20 | residual limits; exceeding steps are logged |
| `--ik-hold-on-error` | off | the offending arm keeps its previous joints on such steps (the next step is warm-started from the held joints, so the step guard stays honoured) |
| `--gravity-comp` | off | `tau_ff = G(q)` from pinocchio RNEA on the URDF (hands included) |
| `--verify-checkpoint DIR` | – | assert `exclude_state=true` and the modality layout of the checkpoint |
| `--log-dir DIR` | – | dump every chunk (measured joints, FK state, server rows, absolute rows, IK trajectory, residuals) as `.npz` |

All smoothing flags of `eval_h2_groot.py` (`--smooth-alpha`, `--boundary-*`)
work the same way, applied to the IK joint targets.

## Log lines to watch

```
server eef output detected as ABSOLUTE  [left: absolute (step0 |p|=0.412 m, |p-p_state|=0.004 m, ...)]
chunk 1: age 210 ms, start_step=6 horizon=16 | IK 32 steps in 3.1 ms | pos err max L/R 0.4/0.3 mm, rot err max L/R 0.2/0.1 deg, max joint step 2.5 deg, clamped 0, held 0 | first executed step 1.1 deg from the last command
```

* `pos err` / `rot err` are FK residuals of the IK solution against the model's
  targets — mm and sub-degree when the target is reachable. Large values mean
  the model asked for something outside the arm's workspace or the wrist
  limits; a few mm past the reach boundary the solver stops instead of trading
  large self-motion for micrometres.
* `clamped > 0` means the joint-step guard engaged (a target that would have
  required > 0.35 rad on one joint within one 33 ms step); `held > 0` means
  `--ik-hold-on-error` kept an arm still on that many steps.
* `first executed step ... from the last command` is the jump from the last
  commanded joints to the new chunk's first executed step; large values indicate
  stale chunks or the model disagreeing with the previous chunk. In async mode a
  chunk older than its whole horizon is dropped (`chunk N dropped: ... ms old`).

## Head-only / relative-hands checkpoints (`run_eval_h2_groot_rel_eef_headonly.sh`, 2026-09-18)

`gr00t-n1.7-h2-sharpa-{tele,cotrain}-eef-camright-headonly-rel-ckpt40000` were
trained on `trocar_tele_eef_camright_0915` (55 teleop episodes of
`trocar_plate_lerobot_0915`) and `trocar_cotrain_eef_camright_raw_0917` (the
same 55 + 151 raw ego episodes). Differences to `trocar_cotrain_rel`:

| | `trocar_cotrain_rel` (`run_eval_h2_groot_rel_eef.sh`) | `*_camright_headonly_rel` (`run_eval_h2_groot_rel_eef_headonly.sh`) |
|---|---|---|
| video keys | `high`, `left_wrist_view`, `right_wrist_view` | `high` only (`--head-only`; wrist cameras not read) |
| `high` | `observation.images.head_right` | same: robot `head_right` DDS stream (`cam_right_high`) -> `--head-eye right` |
| eef | RELATIVE `hand_C_MC`, right-eye camera frame | same (URDF `head_camera_right_optical_frame`; pelvis frame at inference) |
| hands | ABSOLUTE 22 joints | RELATIVE / NON_EEF: `hand[t+k] - hand[t]`; the server composes `state.hand + rel` and returns absolute targets |
| state | dropped (`exclude_state=true` needed) | same |
| head | command 0.7 (`--head-pitch-home 0.7`); FK calibration pitch 0.7 / yaw 0.095 | same command 0.7 (`--head-pitch-home 0.7`); the dataset's camera-frame FK used the measured 0.837 / -0.0125 -> `--cam-head-pitch 0.837 --cam-head-yaw -0.0125` (only matters with `--eef-frame camera`) |
| init posture | `init_post_trocar.yaml` (trocar_test_lerobot, mean of the first 3 frames) | `init_post_trocar_0915.yaml` (`make_init_posture_yaml.py --frame 25` on trocar_plate_lerobot_0915: mean over the 55 episodes of the pose at frame_index 25; arms differ by up to 28 deg from the old one) |

The client code path is unchanged: FK -> `state.*_eef` -> server -> absolute
eef rows -> IK; the hand rows are executed as before because the server already
composed them. The only new requirement is that `state.left_hand/right_hand` are
the *measured* Sharpa joints: `read_hand_state` returns zeros on a read error,
and the script now warns when a connected hand reports all zeros.

`--verify-checkpoint` (set by the launcher when `REL_EEF_CKPT=<served checkpoint
dir>` is exported) also checks the video keys against `--head-only` and accepts
RELATIVE / NON_EEF hand groups; `--cam-head-pitch` is used as the reference for
the "head view differs from training" warning.

Server side (once per checkpoint, before starting `run_gr00t_server.py` as
usual): `python patch_groot_processor_exclude_state.py <checkpoint dir>`. Both
`checkpoint-40000` dirs under `~/Isaac-GR00T/checkpoints/` were patched on
2026-09-18 (`processor_config.json.orig` kept).

Run (tv env):

```bash
bash run_eval_h2_groot_rel_eef_headonly.sh "pick up the trocar and place it in the tray" 300
bash run_eval_h2_groot_rel_eef_headonly.sh "pick up the trocar and place it in the tray" 300 --async-policy --log-dir ~/rel_eef_logs
REL_EEF_CKPT=~/Isaac-GR00T/checkpoints/<...>/checkpoint-40000 bash run_eval_h2_groot_rel_eef_headonly.sh "..." 300   # + processor check
python eval_h2_groot_rel_eef.py --check --head-only --head-eye right --task "..." --network-interface eno2 --camera-network-interface eno2
```

Defaults set by the launcher: `--motion --head-only --head-eye right
--head-pitch-home 0.7 --cam-head-pitch 0.837 --cam-head-yaw -0.0125
--hands-source dds --image-source dds --camera-dds-domain 10
--network-interface eno2 --camera-network-interface eno2 --hold-s 5
--action-horizon 16 --gravity-comp --eef-frame pelvis --server-returns auto
--init-state-yaml init_post_trocar_0915.yaml` (`--no-gravity-comp` switches the
gravity feed-forward off again).

Verified offline on 2026-09-18 (tv env): camera-frame FK of the recorded joints
at head 0.837 / -0.0125 reproduces the dataset eef state to 6e-8; the relative
eef chunk of the dataset equals the pelvis-frame FK chunk of the measured joints
to 1e-7 (frame invariance holds for this recording too); composing on the
camera-frame state or on the pelvis-frame state gives the same pelvis targets
(3e-8) and the same IK joints (7e-6 deg), within 2 deg of the recorded joints;
GR00T's own relative->absolute round trip and the client-side composition agree
to 4e-16; GR00T's NON_EEF hands are exactly `action - state` / `state + rel`.

## Right-hand-only / hand-state checkpoints (`run_eval_h2_groot_right_handstate.sh`, 2026-09-20)

`gr00t-n1.7-h2-sharpa-tele-right-handstate-rel-ckpt40000` and
`gr00t-n1.7-h2-sharpa-cotrain-right-handstate-egohand03-rel-ckpt40000` were
trained on the `*_right` twins of the camright datasets (31-dim: `right_eef`
[0:9], `right_hand` [9:31]; `scripts/make_right_only_dataset.py`) with
`cluster/train_gr00t_*_right_rel_handstate*.sbatch`. Differences to the
head-only models above:

| | `*_camright_headonly_rel` | `*_right_handstate_rel` |
|---|---|---|
| state keys | `left_eef right_eef left_hand right_hand` | `right_eef right_hand` (`--arms right`) |
| state token | dropped (`state_dropout 1.0`, `exclude_state=true`) | **used**: `right_hand` (22 joints) is seen by the model; `right_eef` is zeroed by the processor (`zero_state_keys=["right_eef"]`, `state_dropout 0.1` training-only) -> `exclude_state` must stay **false**, do not patch these checkpoints |
| server code | stock | needs the `zero_state_keys` patch (`isaac-ego-inpaint/cluster/gr00t_zero_state_keys.patch`); `~/Isaac-GR00T` has it, the pre-flight checks the local checkout |
| actions | both arms + hands | `right_eef` RELATIVE/EEF, `right_hand` RELATIVE/NON_EEF only |
| left side | driven | left arm holds the `--init-state-yaml` posture on every step; left hand keeps its init posture |

`state.right_eef` is still sent (pelvis-frame FK) because the server composes
the relative eef chunk onto it; the model never sees it, so the pelvis frame
stays valid. `state.right_hand` is now a real model input: the measured Sharpa
joints must be sane (the all-zero warning matters here).

`--verify-checkpoint` derives the expectations from the checkpoint: no-state
models (`state_dropout_prob >= 1`) need `exclude_state=true`, hand-state models
need `exclude_state=false`; an eef group the model would really see (not in
`zero_state_keys`) forces `--eef-frame camera`; the state-key layout must match
`--arms`.

Run (tv env; server started as usual with `--model-path <dir>/checkpoint-40000`):

```bash
REL_EEF_CKPT=~/Isaac-GR00T/checkpoints/gr00t-n1.7-h2-sharpa-tele-right-handstate-rel-ckpt40000/checkpoint-40000 \
bash run_eval_h2_groot_right_handstate.sh "pick up the trocar and place it in the tray" 300
```

Defaults set by the launcher: the head-only defaults plus `--arms right`.

Verified offline on 2026-09-20 (tv env): the held-arm IK returns exactly the
init joints for the left arm and the same right-arm joints as the two-arm solve
(< 1e-6 deg), within 2 deg of the recorded trajectory; server-mode detection,
absolute-row extraction and the right-only observation (`video.high`,
`state.right_eef`, `state.right_hand`) work with one side; the checkpoint checks
pass for both right-hand checkpoints and reject a wrongly patched
`exclude_state`, a model that would see the eef state in the pelvis frame, and a
mismatched `--arms`.
