#!/usr/bin/env bash
# Replay a fixed-rig trajectory .npz (T, 75 joint_positions) on H2 + Sharpa.
#
# Usage:
#   bash run_replay_npz.sh                      # no gravity comp
#   bash run_replay_npz.sh --gravity-comp       # tau_ff = G(q)
#   NPZ_PATH=/path/to/other.npz bash run_replay_npz.sh --speed 0.5
#
# Playback fps and head pitch default to the values stored in the npz.
# Each run prints a per-joint tracking-error table and saves
# $TRACK_LOG_DIR/replay_track_<npz-stem>_gc_{on,off}_*.npz for compare_replay_tracking.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPZ_PATH="${NPZ_PATH:-$HOME/xr_teleoperate/h2_sharpa_fixed_rig_trajectory.npz}"
TRACK_LOG_DIR="${TRACK_LOG_DIR:-$HOME/replay_tracking_logs}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eno2}"
mkdir -p "$TRACK_LOG_DIR"

python "${SCRIPT_DIR}/replay_h2_fixed_rig_npz.py" \
    --npz-path "$NPZ_PATH" \
    --motion \
    --hands-source dds \
    --network-interface "$NETWORK_INTERFACE" \
    --num-repeats 1 \
    --hold-s 3 \
    --track-log-dir "$TRACK_LOG_DIR" \
    "$@"
