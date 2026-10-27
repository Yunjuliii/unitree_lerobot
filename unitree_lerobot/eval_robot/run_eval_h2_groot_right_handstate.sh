#!/usr/bin/env bash
# Run H2 + Sharpa GR00T policy inference for the RIGHT-HAND-ONLY / hand-state / relative-EEF models
# (gr00t-n1.7-h2-sharpa-tele-right-handstate-rel-ckpt40000, gr00t-n1.7-h2-sharpa-cotrain-right-handstate-egohand03-rel-ckpt40000).
#
# Usage:
#   bash run_eval_h2_groot_right_handstate.sh "<task description>" <total_steps> [extra args...]
#
# Examples:
#   bash run_eval_h2_groot_right_handstate.sh "pick up the trocar and place it in the tray" 300
#   bash run_eval_h2_groot_right_handstate.sh "pick up the trocar and place it in the tray" 300 --async-policy
#   bash run_eval_h2_groot_right_handstate.sh "pick up the trocar and place it in the tray" 300 --policy-port 5556
#   bash run_eval_h2_groot_right_handstate.sh "..." 300 --log-dir ~/rel_eef_logs      # dump every chunk for offline QA
#   # Skip the init-posture step entirely (start each episode from arm-home zeros; the left arm then holds zeros):
#   bash run_eval_h2_groot_right_handstate.sh "..." 300 --init-state-yaml ""
#   # One observation without the server (FK state + head snapshot -> /tmp/check_rel_eef_high.png):
#   python eval_h2_groot_rel_eef.py --check --no-arm --head-only --arms right --head-eye right --task "..." \
#          --network-interface eno2 --camera-network-interface eno2 --verify-checkpoint <checkpoint dir>
#
# What these models expect (isaac-ego-inpaint/dataset/trocar_{tele,cotrain}_eef_camright_*_right/README.md,
# cluster/train_gr00t_*_right_rel_handstate*.sbatch):
#   - video : "high" only = observation.images.head_right = the robot's head_right DDS stream
#             -> --head-only --head-eye right (wrist cameras are neither read nor required)
#   - state : right_eef (hand_C_MC pose, right-eye camera frame) + right_hand (22 Sharpa joints) ONLY (--arms right).
#             The processor ZEROES right_eef in the state token (zero_state_keys) and feeds the real right_hand:
#             the model sees the fingers, never the arm pose. The raw right_eef we send is still the anchor the
#             server composes the RELATIVE eef chunk onto (pelvis frame at inference, frame-invariant).
#             -> exclude_state must stay FALSE for these checkpoints: do NOT run patch_groot_processor_exclude_state.py
#                on them (state_dropout 0.1 was training-time only).
#             -> the GR00T server must run from a gr00t checkout carrying the zero_state_keys patch (~/Isaac-GR00T has it;
#                the REL_EEF_CKPT pre-flight checks that checkout).
#   - action: right_eef RELATIVE/EEF/XYZ_ROT6D, right_hand RELATIVE/NON_EEF -> the server returns absolute targets.
#   - left  : not part of the model. The left arm holds the --init-state-yaml posture on every step and the left
#             hand keeps the init hand posture (not commanded during the episode).
#   - head  : commanded to pitch 0.7 rad like the recording (--head-pitch-home 0.7); 0.837 / -0.0125 are the dataset's
#             camera-frame FK angles (--cam-head-pitch/--cam-head-yaw, only used with --eef-frame camera).
#   - arms  : --gravity-comp on by default (as in run_eval_h2_groot.sh); --no-gravity-comp switches it off.
#
# Prereqs:
#   - sharpa_dds_bridge --side both running on Thor
#   - GR00T policy server reachable at --policy-host:--policy-port (default localhost:5555)
#   - tv conda env active, native DDS camera publisher running on Thor domain 10
#   - robot DDS (domain 0) and camera DDS (domain 10) both over eno2 (--network-interface / --camera-network-interface)
#   - h2_with_sharpa URDF at ~/xr_teleoperate/h2_with_sharpa/urdf/H2_with_sharpa_hands.urdf (or $H2_IK_URDF)
#   - $INIT_STATE_YAML exists (defaults to init_post_trocar_0915.yaml next to this script = mean over the 55
#     trocar_plate_lerobot_0915 episodes of the pose at frame_index 25). Generate with:
#       python make_init_posture_yaml.py ~/isaac-ego-inpaint/dataset/trocar_plate_lerobot_0915 --frame 25 --output <yaml>
#   - optional: REL_EEF_CKPT=<served checkpoint dir> -> exclude_state / zero_state_keys / state+video keys / eef frame
#     are verified before the robot moves
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
# the INIT_STATE_YAML env var, or by passing --init-state-yaml <path> as an extra arg. Also the posture the LEFT arm holds.
INIT_STATE_YAML="${INIT_STATE_YAML:-$SCRIPT_DIR/init_post_trocar_0915.yaml}"
if [[ ! -f "$INIT_STATE_YAML" ]]; then
    echo "WARN: init-state YAML not found at $INIT_STATE_YAML — robot will start from arm zero pose (left arm holds zeros)." >&2
    echo "      Generate one with make_init_posture_yaml.py, or override via" >&2
    echo "      INIT_STATE_YAML=/path/to/file.yaml bash $0 ..." >&2
    INIT_STATE_ARGS=()
else
    INIT_STATE_ARGS=(--init-state-yaml "$INIT_STATE_YAML" --init-state-hold-s 10)
fi

# Optional pre-flight check of the served checkpoint's processor_config.json (exclude_state must be FALSE here,
# right_eef in zero_state_keys, state keys right_eef/right_hand, video key high).
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
    echo "NOTE: REL_EEF_CKPT not set — the served checkpoint's processor_config.json is not verified (exclude_state must be false, zero_state_keys must hold right_eef)." >&2
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
    --arms right \
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
