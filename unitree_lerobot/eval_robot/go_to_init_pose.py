"""
Move H2 arms + Sharpa hands to the inference ready posture and hold it.

Goes straight to the init-state YAML posture used by eval_h2_groot.py (same controller
settings, no cameras, no policy) and holds it until Ctrl+C.

Usage (tv conda env, run from repo root; sharpa_dds_bridge must be running on Thor):
    python unitree_lerobot/eval_robot/go_to_init_pose.py

    # another posture file; go home on exit
    python unitree_lerobot/eval_robot/go_to_init_pose.py \
        --init-state-yaml unitree_lerobot/eval_robot/init_post_trocar_0915.yaml --return-home
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_h2_groot import (  # noqa: E402
    _command_dual_arm, _go_home, _go_to_init_state, _load_init_state_yaml,
    _make_arm_ik, _setup_hands, _teardown_hands, logger,
)

_DEFAULT_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_post_trocar_0921.yaml")


def main():
    ap = argparse.ArgumentParser(description="Move H2 + Sharpa to the inference init posture and hold.")
    ap.add_argument("--init-state-yaml", default=_DEFAULT_YAML)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to hold the posture; 0 = until Ctrl+C")
    ap.add_argument("--hold-s", type=int, default=3, help="hold at home (with --go-home-first / --return-home)")
    ap.add_argument("--init-state-hold-s", type=int, default=3,
                    help="settle time after commanding the posture (default: 3)")
    ap.add_argument("--go-home-first", action="store_true",
                    help="go to home before the posture, like eval_h2_groot (default: straight from current pose)")
    ap.add_argument("--return-home", action="store_true",
                    help="on exit, return arms to home and hands to neutral (default: stay)")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hands", dest="hands", action="store_true", default=True)
    g.add_argument("--no-hands", dest="hands", action="store_false")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--arm", dest="arm", action="store_true", default=True)
    g.add_argument("--no-arm", dest="arm", action="store_false")
    ap.add_argument("--hands-source", choices=["dds", "sdk"], default="dds")
    ap.add_argument("--hand-speed-coeff", type=float, default=0.5)
    ap.add_argument("--hand-current-coeff", type=float, default=0.6)

    # Defaults match the H2 GR00T launchers (run_eval_h2_groot*.sh).
    ap.add_argument("--dds-domain", type=int, default=0)
    ap.add_argument("--network-interface", default="eno2", metavar="IFACE")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--motion", dest="motion", action="store_true", default=True,
                   help="H2 motion mode (rt/arm_sdk), robot standing under WBC (default)")
    g.add_argument("--no-motion", dest="motion", action="store_false")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gravity-comp", dest="gravity_comp", action="store_true", default=True)
    g.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false")
    ap.add_argument("--head-pitch-home", type=float, default=0.7, metavar="RAD")
    ap.add_argument("--kp-low",   type=float, default=None)
    ap.add_argument("--kp-wrist", type=float, default=None)
    ap.add_argument("--kd-low",   type=float, default=None)
    ap.add_argument("--kd-wrist", type=float, default=None)
    args = ap.parse_args()

    init_state = _load_init_state_yaml(args.init_state_yaml)

    if args.arm or (args.hands and args.hands_source == "dds"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(args.dds_domain, args.network_interface)

    left_hand, right_hand = _setup_hands(args)

    # Same order as eval_h2_groot: gravity model before the arm controller starts publishing.
    arm_ik = _make_arm_ik(bool(args.arm and args.gravity_comp))
    arm_ctrl = None
    if args.arm:
        from teleop.robot_control.robot_arm import H2_ArmController
        logger.info(f"Initialising H2_ArmController (motion_mode={args.motion}, "
                    f"head_pitch_home={args.head_pitch_home})...")
        arm_ctrl = H2_ArmController(
            motion_mode=args.motion,
            head_pitch_home=args.head_pitch_home,
            kp_low=args.kp_low, kp_wrist=args.kp_wrist,
            kd_low=args.kd_low, kd_wrist=args.kd_wrist,
        )
        _command_dual_arm(arm_ctrl, arm_ctrl.get_current_dual_arm_q(), arm_ik)
        arm_ctrl.speed_gradual_max()

    try:
        if args.go_home_first:
            _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s, arm_ik=arm_ik)
        _go_to_init_state(arm_ctrl, left_hand, right_hand, init_state,
                          hold_s=args.init_state_hold_s, arm_ik=arm_ik)
        logger.info("At init posture. Holding" +
                    (f" for {args.duration:.0f}s." if args.duration > 0 else " — Ctrl+C to exit."))
        t0 = time.time()
        while args.duration <= 0 or time.time() - t0 < args.duration:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if args.return_home:
            _go_home(arm_ctrl, left_hand, right_hand, hold_s=args.hold_s, arm_ik=arm_ik)
            _teardown_hands(args, left_hand, right_hand)
    logger.info("Done.")


if __name__ == "__main__":
    main()
