# eval_h2_groot.py — H2 + Sharpa GR00T policy inference

Drives the H2 robot with a remote GR00T policy server. Reads cameras, builds
the observation dict, sends it to the policy, executes returned action chunks
on the arms and hands.

## Architecture

```
            ┌─────────────────────────────┐
            │ GR00T policy server         │
            │ (uv env, separate terminal) │
            └──────────────┬──────────────┘
                           │ ZMQ + msgpack
                           ▼
        ┌─────────────────────────────────────┐
        │ eval_h2_groot.py  (tv conda env)    │
        │                                     │
        │  cameras  → obs → policy → action   │
        │     │                       │       │
        │     │                       ▼       │
        │     │                  arm + hands  │
        │     ▼                       │       │
        │  ROSImageClient             ▼       │
        │  (CycloneDDS, 4 topics)   H2_ArmCtrl │
        │                           SharpaDDS  │
        └─────────────────────────────────────┘
                       │
                       ▼ DDS (rt/lowcmd or rt/arm_sdk + rt/sharpa/*/cmd)
                       ▼
                  H2 robot + Sharpa hands
```

- **Cameras**: ROS 2 (CycloneDDS) topics from `sensing_h2_ros_gstreamer` on
  Thor — four streams: head left/right and wrist left/right. Legacy ZMQ
  teleimager source still works via `--image-source zmq`.
- **Arm**: `H2_ArmController` from xr_teleoperate. Publishes to `rt/lowcmd`
  in debug mode or `rt/arm_sdk` with `--motion` (required when the robot is
  standing on its legs).
- **Hands**: `SharpaHandDDSClient` from xr_teleoperate publishes `HandCmd_`
  on `rt/sharpa/{left,right}/cmd`. The C++ `sharpa_dds_bridge` on Thor owns
  the SDK side. Direct SDK is still available via `--hands-source sdk`.

## Prerequisites (in order)

1. **Boot the robot** and bring up cameras, hands, and standing posture as
   described in `~/binliu/xr_teleoperate/teleop/docs/startup_procedure.md`
   (steps 1–5).
2. **Start the Sharpa DDS bridge on Thor:**
   ```bash
   ssh unitree@192.168.123.163
   cd ~/Sharpa/bridge_dds && ./sharpa_dds_bridge --side both
   ```
3. **Start the GR00T policy server** (in GR00T's uv env, separate terminal):
   ```bash
   cd /home/nvidia/binliu/Isaac-GR00T
   uv run python gr00t/eval/run_gr00t_server.py \
       --model-path /path/to/checkpoint \
       --embodiment-tag new_embodiment \
       --port 5555
   ```
4. **Source ROS and activate the tv env on the workstation:**
   ```bash
   source /opt/ros/jazzy/setup.bash
   conda activate tv
   cd ~/binliu/unitree_lerobot
   ```

## Observation schema

What the script sends to the policy each step. Shapes are after adding the
`(B=1, T=1)` leading dims expected by GR00T.

**Single-eye head mode** (`--head-eye left` or `--head-eye right`, default):

| Key                                          | Shape          | dtype |
|----------------------------------------------|----------------|-------|
| `video.high`                                 | (1,1,H,W,3)    | uint8 |
| `video.left_wrist_view`                      | (1,1,H,W,3)    | uint8 |
| `video.right_wrist_view`                     | (1,1,H,W,3)    | uint8 |
| `state.left_arm`                             | (1,1,7)        | float32 |
| `state.right_arm`                            | (1,1,7)        | float32 |
| `state.left_hand`                            | (1,1,22)       | float32 |
| `state.right_hand`                           | (1,1,22)       | float32 |
| `language.annotation.human.task_description` | [[str]]        | str   |

**Stereo head mode** (`--head-eye both`):

`video.high` is replaced by:

| Key                          | Shape          | dtype |
|------------------------------|----------------|-------|
| `video.left_eye_view`        | (1,1,H,W,3)    | uint8 |
| `video.right_eye_view`       | (1,1,H,W,3)    | uint8 |

(Wrists and state are unchanged.)

## Action schema

The policy returns an action chunk; the script consumes up to
`--action-horizon` steps before re-querying.

| Key          | Shape    | Unit    |
|--------------|----------|---------|
| `left_arm`   | (B,T,7)  | radians |
| `right_arm`  | (B,T,7)  | radians |
| `left_hand`  | (B,T,22) | radians |
| `right_hand` | (B,T,22) | radians |

## CLI flags (selected)

| Flag                       | Default        | Notes |
|----------------------------|----------------|-------|
| `--policy-host` / `--policy-port` | `localhost` / `5555` | GR00T server location |
| `--task "..."`             | `""`           | Language instruction |
| `--image-source {ros,zmq}` | `ros`          | Camera source |
| `--head-eye {left,right,both}` | `left`     | Which head eye(s) to send |
| `--hands-source {dds,sdk}` | `dds`          | DDS bridge or direct SDK |
| `--motion`                 | off            | `rt/arm_sdk` mode for standing |
| `--head-pitch-home RAD`    | `0.6`          | Held throughout inference |
| `--dds-domain N`           | `0`            | Match the bridge / arm publisher |
| `--frequency HZ`           | `30`           | Per-step control rate |
| `--action-horizon N`       | `16`           | Steps to consume per chunk |
| `--steps N`                | `30`           | Total steps per episode |
| `--hold-s S`               | `20`           | Home hold before/between/after |
| `--cam-warmup-s S`         | `5`            | Per-camera first-frame timeout |
| `--no-arm`, `--no-hands`   | both on        | Disable subsystem |
| `--check`                  | off            | Dump observation + camera PNGs to `/tmp/` |

## Common commands

### Standing inference (recommended workflow)

```bash
python unitree_lerobot/eval_robot/eval_h2_groot.py \
    --policy-host localhost --policy-port 5555 \
    --task "pick up the apple" \
    --motion \
    --head-eye left \
    --hold-s 3
```

### Quick check mode — no policy server needed

Validates camera flow, hand connectivity (via DDS bridge), and arm controller
init. Saves one snapshot per camera to `/tmp/check_<key>.png` so you can
verify what the policy would see.

```bash
python unitree_lerobot/eval_robot/eval_h2_groot.py --check --task "test"
```

### Stereo head mode

If your policy was trained with two head views:

```bash
python unitree_lerobot/eval_robot/eval_h2_groot.py \
    --policy-host localhost --task "..." \
    --head-eye both --motion
```

### Right-eye only

```bash
python unitree_lerobot/eval_robot/eval_h2_groot.py \
    --policy-host localhost --task "..." \
    --head-eye right --motion
```

### Arm only — no Sharpa hardware required

```bash
python unitree_lerobot/eval_robot/eval_h2_groot.py \
    --policy-host localhost --task "..." \
    --no-hands --motion
```

### Legacy ZMQ image server

```bash
python unitree_lerobot/eval_robot/eval_h2_groot.py \
    --policy-host localhost --task "..." \
    --image-source zmq --img-server-ip 192.168.124.162
```

### Direct Sharpa SDK (stop the bridge first)

```bash
# on Thor: killall sharpa_dds_bridge
python unitree_lerobot/eval_robot/eval_h2_groot.py \
    --policy-host localhost --task "..." \
    --hands-source sdk
```

## Notes on the GR00T protocol

The script speaks the same wire protocol as `gr00t.policy.server_client`
(ZMQ REQ/REP, msgpack-packed nested dicts with custom numpy encoding). No
dependency on the `gr00t` package — only `zmq`, `msgpack`, `numpy` are used.

`GR00TClient` exposes three endpoints:

- `ping()` — returns `True` if the server responds.
- `reset()` — call once per episode boundary; resets the policy's internal
  state (e.g. action queues).
- `get_action(obs)` — returns `(action_chunk, info)`.

## Inference loop shape

```
init → home + hold(--hold-s)
for episode in 1..∞:
    client.reset()
    while step_count < --steps:
        obs = get_observations()
        chunk = client.get_action(obs)
        for t in 0..min(--action-horizon, chunk_len, remaining):
            send arm + hands at frame t
            sleep 1/--frequency
    home + hold(--hold-s)
```

Ctrl+C at any point triggers a final home + hold and clean teardown of arm,
hands, and image client.

## Troubleshooting

- **`video` dict empty in `--check` output** — cameras aren't publishing.
  Verify with `ros2 topic hz /head/left/image_raw` (ROS source) or check
  `enable_zmq` in the teleimager server's cam config (ZMQ source).
- **Hand DDS client returns zeros for state** — the C++ bridge isn't running
  on Thor, or it's running on a different DDS domain than `--dds-domain`.
- **"Tactile sensor not ready" on the bridge side** — see
  `~/binliu/xr_teleoperate/teleop/docs/sharpa_hands_thor.md`.
- **Arm doesn't move in motion mode** — the WBC isn't yielding because the
  robot isn't in a state where motion mode is accepted. Confirm step 5 of
  the startup procedure (R2+Y to stand) completed.
- **Crash on first frame about `bgr` being `None`** — increase `--cam-warmup-s`
  or check that ROS 2 is sourced (`echo $RMW_IMPLEMENTATION` should show
  `rmw_cyclonedds_cpp`).

## Related files

- `eval_h2.py` — earlier, simpler eval script (no GR00T server, no DDS bridge).
- `replay_h2.py` — replay a recorded LeRobot episode on the same hardware
  setup (same arm / hand abstractions, no policy server).
- `~/binliu/xr_teleoperate/teleop/utils/ros_image_client.py` — the ROS camera
  client this script imports.
- `~/binliu/xr_teleoperate/teleop/robot_control/robot_hand_sharpa_dds.py` —
  the DDS hand client.
- `~/binliu/xr_teleoperate/teleop/robot_control/robot_arm.py` —
  `H2_ArmController` (motion mode, head control, velocity clipping).
- `~/binliu/xr_teleoperate/scripts/sharpa_dds_bridge.cpp` — the C++ bridge on
  Thor that owns the SDK side and bridges to DDS.
