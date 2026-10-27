#!/usr/bin/env bash
# Replay the 58-dim observation.h2_joints column (arm IK + Sharpa hands) of an ego
# LeRobot dataset on H2 + Sharpa, with gravity compensation (tau_ff = G(q)).
#
# Usage (tv conda env):
#   DATASET_DIR=/path/to/lerobot_dataset bash run_replay_lerobot_column.sh                  # episode 0, gravity comp on
#   DATASET_DIR=... bash run_replay_lerobot_column.sh --speed 0.5                              # half speed
#   DATASET_DIR=... bash run_replay_lerobot_column.sh --confirm --num-repeats 3                # Enter before each replay
#   DATASET_DIR=... bash run_replay_lerobot_column.sh --no-gravity-comp                        # without tau_ff
#   DATASET_DIR=... bash run_replay_lerobot_column.sh --episodes 2
#
# Each run prints a per-joint tracking-error table (measured - target) and saves
# $TRACK_LOG_DIR/replay_track_<dataset>_ep<N>_gc_{on,off}_rep<K>_*.npz for compare_replay_tracking.py.
#
# Prereqs: sharpa_dds_bridge --side both running on Thor; robot standing in WBC (--motion).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${DATASET_DIR:-}" ]]; then
    echo "ERROR: set DATASET_DIR=/path/to/lerobot_dataset" >&2
    exit 1
fi
TRACK_LOG_DIR="${TRACK_LOG_DIR:-$HOME/replay_tracking_logs}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-eno2}"
mkdir -p "$TRACK_LOG_DIR"

cd "$REPO_ROOT"
python "${SCRIPT_DIR}/replay_h2_lerobot_column.py" \
    --dataset-root "$DATASET_DIR" \
    --column observation.h2_joints \
    --episodes 0 \
    --gravity-comp \
    --motion \
    --head-pitch-home 0.7 \
    --hands-source dds \
    --network-interface "$NETWORK_INTERFACE" \
    --num-repeats 1 \
    --hold-s 3 \
    --track-log-dir "$TRACK_LOG_DIR" \
    "$@"
