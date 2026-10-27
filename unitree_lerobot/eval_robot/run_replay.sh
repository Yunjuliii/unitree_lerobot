#!/usr/bin/env bash
# Replay one episode of a LeRobot dataset on H2 + Sharpa.
#
# Usage:
#   bash run_replay.sh                      # episode 0, no gravity comp
#   bash run_replay.sh --gravity-comp       # same episode with tau_ff = G(q)
#   bash run_replay.sh --episode 3 --speed 0.5
#
# Each run prints a per-joint tracking-error table (measured - target) and saves
# the raw log to $TRACK_LOG_DIR/replay_track_ep<N>_gc_{on,off}_*.npz, so an
# on/off pair can be compared with compare_replay_tracking.py.
#
# Prereqs: sharpa_dds_bridge --side both running on Thor; robot in WBC (--motion).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${DATASET_DIR:-$HOME/xr_teleoperate/dataset/trocar_plate_lerobot_v21}"
TRACK_LOG_DIR="${TRACK_LOG_DIR:-$HOME/replay_tracking_logs}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eno2}"
mkdir -p "$TRACK_LOG_DIR"

python "${SCRIPT_DIR}/replay_h2.py" \
    --dataset-dir "$DATASET_DIR" \
    --episode 0 \
    --motion \
    --hands-source dds \
    --network-interface "$NETWORK_INTERFACE" \
    --num-repeats 1 \
    --hold-s 3 \
    --track-log-dir "$TRACK_LOG_DIR" \
    "$@"
