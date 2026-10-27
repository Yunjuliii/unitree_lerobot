#!/usr/bin/env bash
# Run H2 + Sharpa GR00T inference for the RELATIVE-EEF / no-state model
# (trocar_cotrain_rel, cluster/train_gr00t_cotrain_rel.sbatch).
#
# Usage:
#   bash run_eval_h2_groot_rel_eef.sh "<task description>" <total_steps> [extra args...]
#
# Examples:
#   bash run_eval_h2_groot_rel_eef.sh "pick up the trocar and place it in the tray" 300
#   bash run_eval_h2_groot_rel_eef.sh "pick up the trocar and place it in the tray" 300 --async-policy
#   bash run_eval_h2_groot_rel_eef.sh "..." 300 --log-dir ~/rel_eef_logs        # dump every chunk for offline QA
#   bash run_eval_h2_groot_rel_eef.sh "..." 300 --init-state-yaml ""            # start from arm zeros
#   python eval_h2_groot_rel_eef.py --check --task "..." --head-eye right       # one observation, no server
#
# Pipeline (see eval_h2_groot_rel_eef.py): arm joints -> FK hand_C_MC (pelvis
# frame; the training data used the right-eye camera frame but relative poses
# are frame-invariant, --eef-frame camera switches back) -> state.{left,right}_eef
# -> server composes the relative chunk onto it -> absolute poses -> sequential
# IK -> joint targets at 30 Hz.  Hands are absolute 22-joint targets as before.
#
# Prereqs:
#   - sharpa_dds_bridge --side both running on Thor
#   - GR00T policy server on --policy-host:--policy-port (default localhost:5555):
#         cd ~/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
#             --model-path <ckpt_trocar_cotrain_rel_sd1.0_bs1024/checkpoint-XXXXX> \
#             --embodiment-tag new_embodiment --port 5555
#     The checkpoint's processor_config.json MUST have "exclude_state": true
#     (the model was trained with state_dropout 1.0 and must not see the state):
#         python patch_groot_processor_exclude_state.py <checkpoint dir>        # once, before starting the server
#   - tv conda env active, native DDS camera publisher on Thor (domain 10)
#   - NIC: pass --network-interface <NIC to the robot, e.g. eno1> (robot DDS, domain 0) and
#     --camera-network-interface <NIC> (camera DDS, domain 10); with only the camera one given the
#     robot DDS reuses it. Without either, CycloneDDS may auto-select docker0/wifi and the arm
#     controller loops on "Waiting to subscribe dds..." forever.
#   - h2_with_sharpa URDF: $H2_IK_URDF (default ~/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf)
#   - $INIT_STATE_YAML (default: init_post_trocar.yaml next to this script = mean of the
#     first 3 frames of the 101 trocar teleop episodes)
#   - Wrist cameras: trocar_cotrain_rel's teleop half carries the wrist labels of the recording
#     session (the "wrist_swaped" datasets exist because those were once found mislabelled).
#     Check once which labelling the live robot matches:
#         python eval_h2_groot_rel_eef.py --check --task "..." --reference-dataset ~/isaac-ego-inpaint/dataset/trocar_cotrain_rel
#     and add --swap-wrist-cams if the live left/right wrist views are mirrored w.r.t. the dataset frames.
#
# Override any default by passing the matching flag as an extra argument:
#   --policy-host, --policy-port, --head-eye, --hold-s, --action-horizon, --frequency,
#   --init-state-yaml, --init-state-hold-s, --dds-domain, --camera-dds-domain,
#   --camera-network-interface, --network-interface, --gravity-comp, --swap-wrist-cams,
#   --server-returns {auto,absolute,relative}, --max-joint-step-rad, --ik-hold-on-error,
#   --eef-frame {pelvis,camera} (+ --cam-head-pitch/--cam-head-yaw/--cam-head-from-robot),
#   --verify-checkpoint <dir>, --log-dir
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"<task description>\" <total_steps> [extra args...]" >&2
    exit 1
fi

TASK="$1"
STEPS="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# URDF with the hand_C_MC + head camera frames (same asset the training poses were computed from).
# $H2_IK_URDF wins; otherwise the usual checkout locations are tried (xr_teleoperate next to this
# unitree_lerobot checkout, ~/workspace/xr_teleoperate, ~/xr_teleoperate, ~/h2_with_sharpa).
REPO_PARENT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
URDF_CANDIDATES=(
    "${H2_IK_URDF:-}"
    "${XR_TELEOPERATE_ROOT:-/nonexistent}/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"
    "${REPO_PARENT}/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"
    "$HOME/workspace/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"
    "$HOME/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"
    "$HOME/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"
    "$HOME/isaac-ego-inpaint/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf"
)
H2_IK_URDF=""
for cand in "${URDF_CANDIDATES[@]}"; do
    if [[ -n "$cand" && -f "$cand" ]]; then H2_IK_URDF="$cand"; break; fi
done
if [[ -z "$H2_IK_URDF" ]]; then
    echo "ERROR: H2_with_sharpa_hands.urdf not found; set H2_IK_URDF=/path/to/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf" >&2
    exit 1
fi
export H2_IK_URDF
echo "URDF: $H2_IK_URDF"

# Init posture: mean of the first frames of the trocar teleop data (generated by
# validate_h2_eef_kinematics.py --write-init-yaml). Override with INIT_STATE_YAML=... or --init-state-yaml <path>.
INIT_STATE_YAML="${INIT_STATE_YAML:-}"
if [[ -z "$INIT_STATE_YAML" ]]; then
    INIT_STATE_ARGS=()
elif [[ ! -f "$INIT_STATE_YAML" ]]; then
    echo "WARN: init-state YAML not found at $INIT_STATE_YAML — robot will start from arm zero pose." >&2
    INIT_STATE_ARGS=()
else
    INIT_STATE_ARGS=(--init-state-yaml "$INIT_STATE_YAML" --init-state-hold-s 10)
fi

# Checkpoint directory the server is serving, if it is readable from this host:
#   REL_EEF_CKPT=/home/nvidia/yunju_model/ckpt_trocar_cotrain_rel_.../checkpoint-XXXXX bash run_eval_h2_groot_rel_eef.sh ...
# -> the eval script asserts exclude_state=true / use_relative_action=true / modality layout before moving the robot.
VERIFY_ARGS=()
if [[ -n "${REL_EEF_CKPT:-}" ]]; then
    VERIFY_ARGS=(--verify-checkpoint "$REL_EEF_CKPT")
else
    echo "NOTE: REL_EEF_CKPT not set — the checkpoint's processor_config.json is not verified (exclude_state must be true)." >&2
fi

python "${SCRIPT_DIR}/eval_h2_groot_rel_eef.py" \
    --policy-host localhost \
    --policy-port 5555 \
    --task "${TASK}" \
    --steps "${STEPS}" \
    --motion \
    --head-eye right \
    --head-pitch-home 0.7 \
    --hands-source dds \
    --image-source dds \
    --camera-dds-domain 10 \
    --hold-s 20 \
    --action-horizon 16 \
    --eef-frame pelvis \
    --server-returns auto \
    "${INIT_STATE_ARGS[@]}" \
    "${VERIFY_ARGS[@]}" \
    "$@"
