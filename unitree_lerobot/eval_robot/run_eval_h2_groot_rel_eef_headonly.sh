#!/usr/bin/env bash
# Run H2 + Sharpa GR00T policy inference for the head-only / relative-EEF / relative-hands / no-state models
# (gr00t-n1.7-h2-sharpa-{tele,cotrain}-eef-camright-headonly-rel-ckpt40000).
#
# Usage:
#   bash run_eval_h2_groot_rel_eef_headonly.sh "<task description>" <total_steps> [extra args...]
#
# Examples:
#   bash run_eval_h2_groot_rel_eef_headonly.sh "pick up the trocar and place it in the tray" 300
#   bash run_eval_h2_groot_rel_eef_headonly.sh "pick up the trocar and place it in the tray" 300 --async-policy
#   bash run_eval_h2_groot_rel_eef_headonly.sh "pick up the trocar and place it in the tray" 300 --policy-port 5556
#   bash run_eval_h2_groot_rel_eef_headonly.sh "..." 300 --log-dir ~/rel_eef_logs      # dump every chunk for offline QA
#   # Skip the init-posture step entirely (start each episode from arm-home zeros):
#   bash run_eval_h2_groot_rel_eef_headonly.sh "..." 300 --init-state-yaml ""
#   # One observation without the server (FK state + head snapshot -> /tmp/check_rel_eef_high.png):
#   python eval_h2_groot_rel_eef.py --check --head-only --head-eye right --task "..." \
#          --network-interface eno2 --camera-network-interface eno2
#
# What these models expect (isaac-ego-inpaint/dataset/trocar_{tele,cotrain}_eef_camright_*/README.md):
#   - video : "high" only = observation.images.head_right = the robot's head_right DDS stream
#             -> --head-only --head-eye right (wrist cameras are neither read nor required)
#   - eef   : hand_C_MC pose of each hand in the right-eye camera frame, RELATIVE (T_state^-1 T_action).
#             The relative pose is the same in any frame fixed to the pelvis, so the pelvis frame is used at
#             inference (--eef-frame pelvis; --eef-frame camera gives identical targets, see eval_h2_groot_rel_eef.md)
#   - hands : 22 Sharpa joints per side, RELATIVE (hand[t+k] - hand[t]); the server composes them onto the
#             measured hand state and returns absolute joint targets
#   - state : trained with state_dropout 1.0 -> the served checkpoint MUST have exclude_state=true in its
#             processor_config.json (once, before starting the server):
#                 python patch_groot_processor_exclude_state.py <checkpoint dir>
#   - head  : the head is COMMANDED to pitch 0.7 rad, the same command as during the teleop recording
#             (--head-pitch-home 0.7). The 0.837 / -0.0125 rad in the dataset README are the head joint values the
#             dataset builder used for the camera-frame FK; they only enter --cam-head-pitch/--cam-head-yaw, which
#             matter for --eef-frame camera alone (pelvis frame is the default).
#   - arms  : --gravity-comp on by default (as in run_eval_h2_groot.sh): tau_ff = G(q) from pinocchio RNEA on the
#             h2_with_sharpa URDF incl. the Sharpa hand masses; pass --no-gravity-comp to run with tau_ff = 0
#
# Prereqs:
#   - sharpa_dds_bridge --side both running on Thor
#   - GR00T policy server reachable at --policy-host:--policy-port (default localhost:5555)
#   - tv conda env active, native DDS camera publisher running on Thor domain 10
#   - robot DDS (domain 0) and camera DDS (domain 10) both over eno2 (--network-interface / --camera-network-interface)
#   - h2_with_sharpa URDF at ~/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf (or $H2_IK_URDF)
#   - $INIT_STATE_YAML exists (defaults to init_post_trocar_0915.yaml next to this script = mean over the 55
#     trocar_plate_lerobot_0915 episodes of the pose at frame_index 25, the recording the models were trained on).
#     Generate with:
#       python make_init_posture_yaml.py ~/isaac-ego-inpaint/dataset/trocar_plate_lerobot_0915 --frame 25 --output <yaml>
#   - optional: REL_EEF_CKPT=<served checkpoint dir> -> exclude_state / use_relative_action / modality layout are
#     verified before the robot moves
#
# Override any default by passing the matching flag as an extra argument:
#   --policy-host, --policy-port, --head-eye, --head-pitch-home, --hold-s, --action-horizon,
#   --init-state-yaml, --init-state-hold-s, --frequency, --dds-domain, --camera-dds-domain,
#   --camera-network-interface, --network-interface, --no-gravity-comp, --eef-frame {pelvis,camera},
#   --server-returns {auto,absolute,relative}, --max-joint-step-rad, --ik-hold-on-error, --async-policy, --log-dir
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 \"<task description>\" <total_steps> [extra args...]" >&2
    exit 1
fi

TASK="$1"
STEPS="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Init posture: frame-25 average of the trocar_plate_lerobot_0915 recording (make_init_posture_yaml.py). Override with
# the INIT_STATE_YAML env var, or by passing --init-state-yaml <path> as an extra arg.
INIT_STATE_YAML="${INIT_STATE_YAML:-$SCRIPT_DIR/init_post_trocar_0915.yaml}"
if [[ ! -f "$INIT_STATE_YAML" ]]; then
    echo "WARN: init-state YAML not found at $INIT_STATE_YAML — robot will start from arm zero pose." >&2
    echo "      Generate one with make_init_posture_yaml.py, or override via" >&2
    echo "      INIT_STATE_YAML=/path/to/file.yaml bash $0 ..." >&2
    INIT_STATE_ARGS=()
else
    INIT_STATE_ARGS=(--init-state-yaml "$INIT_STATE_YAML" --init-state-hold-s 10)
fi

# Optional pre-flight check of the served checkpoint's processor_config.json (exclude_state must be true).
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
    echo "NOTE: REL_EEF_CKPT not set — the served checkpoint's processor_config.json is not verified (exclude_state must be true)." >&2
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
    --head-only \
    --head-eye right \
    --head-pitch-home 0.7 \
    --cam-head-pitch 0.837 \
    --cam-head-yaw -0.0125 \
    --hands-source dds \
    --image-source dds \
    --camera-dds-domain 10 \
    --network-interface eno2 \
    --camera-network-interface eno2 \
    --hold-s 5 \
    --action-horizon 16 \
    "${GRAVITY_ARGS[@]}" \
    --eef-frame pelvis \
    --server-returns auto \
    "${INIT_STATE_ARGS[@]}" \
    "${VERIFY_ARGS[@]}" \
    "$@"
