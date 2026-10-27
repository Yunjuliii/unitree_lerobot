#!/usr/bin/env bash
# Run H2 + Sharpa GR00T policy inference for the *_eef_camright_0921_right checkpoints (same processor layout):
#   gr00t-trocar-tele108-eef-camright-0921-right           (dataset: trocar_tele108_eef_camright_0921_right)
#   gr00t-trocar-cotrain-tele108-ego107-eef-camright-0921-right (dataset: trocar_cotrain_tele108_ego107_eef_camright_0921_right)
#
# Usage:
#   bash run_eval_h2_groot_camright_0921_right.sh "<task description>" <total_steps> [extra args...]
#
# Examples:
#   bash run_eval_h2_groot_camright_0921_right.sh "pick up the trocar and place it in the tray" 300
#   bash run_eval_h2_groot_camright_0921_right.sh "..." 300 --async-policy
#   bash run_eval_h2_groot_camright_0921_right.sh "..." 300 --log-dir ~/camright_0921_logs   # dump every chunk for offline QA
#   # One observation without the server (FK state + head/right-wrist snapshot, no robot motion):
#   python eval_h2_groot_rel_eef.py --check --no-arm --arms right --right-wrist-only --swap-wrist-cams \
#          --eef-frame camera --cam-head-pitch 0.837 --cam-head-yaw -0.0125 --head-eye right --task "..." \
#          --network-interface eno2 --camera-network-interface eno2 \
#          --verify-checkpoint /home/yunju/Isaac-GR00T/checkpoints/gr00t-trocar-tele108-eef-camright-0921-right
#
# What these checkpoints expect (isaac-ego-inpaint/dataset/trocar_tele108_eef_camright_0921_right/README.md):
#   - video : observation.images.head_right -> video.high, observation.images.wrist_right -> video.right_wrist_view
#             (no left_wrist_view) -> --arms right --right-wrist-only (head-eye right; wrist cameras ARE read/required,
#             unlike the head-only / hand-state models).
#   - state : right_eef (hand_C_MC pose, RIGHT-EYE CAMERA frame) + right_hand (22 Sharpa joints), BOTH seen by the
#             model (exclude_state=false, not zeroed) -> --eef-frame camera with --cam-head-pitch/--cam-head-yaw set
#             to the dataset's FK calibration angles (0.837 / -0.0125), NOT the head-pitch-home command (0.7 rad,
#             see --head-pitch-home below) -- those are the measured joint values at recording time, the eef math
#             being frame-invariant is what lets --head-pitch-home differ from them without changing the state math
#             (it only changes what the physical camera sees).
#   - action: right_eef RELATIVE/EEF/XYZ_ROT6D, right_hand RELATIVE/NON_EEF -> the server returns absolute targets.
#   - left  : not part of the model. The left arm holds the --init-state-yaml posture on every step and the left
#             hand keeps the init hand posture (not commanded during the episode).
#   - head  : commanded to pitch 0.7 rad like the recording (--head-pitch-home 0.7).
#   - arms  : --gravity-comp on by default (as in run_eval_h2_groot.sh); --no-gravity-comp switches it off.
#
# Prereqs:
#   - sharpa_dds_bridge --side both running on Thor
#   - GR00T policy server reachable at --policy-host:--policy-port (default localhost:5555), started by you, e.g.:
#         cd ~/Isaac-GR00T && uv run python gr00t/eval/run_gr00t_server.py \
#             --model-path /home/yunju/Isaac-GR00T/checkpoints/gr00t-trocar-tele108-eef-camright-0921-right/checkpoint-40000 \
#             --embodiment-tag new_embodiment --port 5555
#   - tv conda env active, native DDS camera publisher running on Thor domain 10
#   - robot DDS (domain 0) and camera DDS (domain 10) both over eno2 (--network-interface / --camera-network-interface)
#   - h2_with_sharpa URDF at ~/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf (or $H2_IK_URDF)
#   - the live robot's wrist cameras are physically left/right-reversed (as of 2026-09-16) -> --swap-wrist-cams is
#     passed below so video.right_wrist_view is sourced from the physical stream that matches the dataset's
#     wrist_right column; drop it if the wiring has been fixed since (verify with --check, see /tmp/check_*.png)
#   - $INIT_STATE_YAML exists (defaults to init_post_trocar_0921.yaml next to this script = mean, over the 108
#     episodes of trocar_tele_0921, of the pose at frame_index 20). Regenerate with:
#       python make_init_posture_yaml.py ~/isaac-ego-inpaint/dataset/trocar_tele_0921 --frame 20 --output <yaml>
#   - optional: REL_EEF_CKPT=<served checkpoint dir> -> exclude_state / state+video keys / eef frame are verified
#     before the robot moves
#
# Override any default by passing the matching flag as an extra argument:
#   --policy-host, --policy-port, --head-eye, --head-pitch-home, --hold-s, --action-horizon,
#   --init-state-yaml, --init-state-hold-s, --frequency, --dds-domain, --camera-dds-domain,
#   --camera-network-interface, --network-interface, --no-gravity-comp, --server-returns {auto,absolute,relative},
#   --max-joint-step-rad, --ik-hold-on-error, --async-policy, --log-dir
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"<task description>\" <total_steps> [extra args...]" >&2
    exit 1
fi

TASK="$1"
STEPS="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Init posture: frame-20 average of the trocar_tele_0921 recording (make_init_posture_yaml.py). Override with
# the INIT_STATE_YAML env var, or by passing --init-state-yaml <path> as an extra arg. Also the posture the LEFT arm holds.
INIT_STATE_YAML="${INIT_STATE_YAML:-$SCRIPT_DIR/init_post_trocar_0921.yaml}"
if [[ ! -f "$INIT_STATE_YAML" ]]; then
    echo "WARN: init-state YAML not found at $INIT_STATE_YAML — robot will start from arm zero pose (left arm holds zeros)." >&2
    echo "      Generate one with make_init_posture_yaml.py, or override via" >&2
    echo "      INIT_STATE_YAML=/path/to/file.yaml bash $0 ..." >&2
    INIT_STATE_ARGS=()
else
    INIT_STATE_ARGS=(--init-state-yaml "$INIT_STATE_YAML" --init-state-hold-s 10)
fi

# Optional pre-flight check of the served checkpoint's processor_config.json (exclude_state must be false,
# state keys right_eef/right_hand, video keys high/right_wrist_view, eef frame camera).
VERIFY_ARGS=()
if [[ -n "${REL_EEF_CKPT:-}" ]]; then
    REL_EEF_CKPT="${REL_EEF_CKPT%/}"
    # A repo dir holding checkpoint-XXXXX subdirs -> use the latest one (the dir with processor_config.json).
    if [[ -d "$REL_EEF_CKPT" && ! -f "$REL_EEF_CKPT/processor_config.json" && ! -f "$REL_EEF_CKPT/processor/processor_config.json" ]]; then
        LATEST_SUB="$(ls -1d "$REL_EEF_CKPT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n 1 || true)"
        if [[ -n "$LATEST_SUB" ]]; then
            echo "REL_EEF_CKPT: using $LATEST_SUB"
            REL_EEF_CKPT="$LATEST_SUB"
        fi
    fi
    VERIFY_ARGS=(--verify-checkpoint "$REL_EEF_CKPT")
else
    echo "NOTE: REL_EEF_CKPT not set — the served checkpoint's processor_config.json is not verified. Set e.g." >&2
    echo "      REL_EEF_CKPT=/home/yunju/Isaac-GR00T/checkpoints/gr00t-trocar-tele108-eef-camright-0921-right bash $0 ..." >&2
fi

# Gravity feed-forward on by default (as in run_eval_h2_groot.sh). --gravity-comp / --no-gravity-comp exclude each
# other in argparse, so the default is dropped when --no-gravity-comp is given as an extra argument.
GRAVITY_ARGS=(--gravity-comp)
for arg in "$@"; do
    if [[ "$arg" == "--no-gravity-comp" ]]; then GRAVITY_ARGS=(); fi
done

python "${SCRIPT_DIR}/eval_h2_groot_rel_eef.py" \
    --policy-host localhost \
    --policy-port 5555 \
    --task "${TASK}" \
    --steps "${STEPS}" \
    --motion \
    --arms right \
    --right-wrist-only \
    --head-eye right \
    --head-pitch-home 0.7 \
    --eef-frame camera \
    --cam-head-pitch 0.837 \
    --cam-head-yaw -0.0125 \
    --hands-source dds \
    --image-source dds \
    --camera-dds-domain 10 \
    --network-interface eno2 \
    --camera-network-interface eno2 \
    --hold-s 5 \
    --action-horizon 16 \
    --swap-wrist-cams \
    "${GRAVITY_ARGS[@]}" \
    --server-returns auto \
    "${INIT_STATE_ARGS[@]}" \
    "${VERIFY_ARGS[@]}" \
    "$@"
