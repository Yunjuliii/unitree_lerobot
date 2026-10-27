"""
Read-only monitor for Sharpa hand joint angles (22 DoF per hand).

Subscribes to the sharpa_dds_bridge state topics on Thor and prints the joint
angles. It never creates a publisher, so no command is ever sent to the hands.

  rt/sharpa/left/state   HandState_  — 22 x motor_state[i].q (degrees)
  rt/sharpa/right/state  HandState_  — 22 x motor_state[i].q (degrees)

Usage:
    python read_sharpa_state.py --network-interface eno2
    python read_sharpa_state.py --network-interface eno2 --side right --unit rad --hz 5
    python read_sharpa_state.py --network-interface eno2 --save sharpa_state.npz
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np

_XR_ROOT = os.path.expanduser("~/xr_teleoperate")
if _XR_ROOT not in sys.path:
    sys.path.insert(0, _XR_ROOT)

SHARPA_DOF = 22
TOPIC_STATE = {"left": "rt/sharpa/left/state", "right": "rt/sharpa/right/state"}


class SharpaStateReader:
    """Subscribe-only reader of one Sharpa hand's joint state (degrees)."""

    def __init__(self, side: str):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

        self.side = side
        self._lock = threading.Lock()
        self._deg = None
        self._stamp = 0.0
        self._count = 0
        self._sub = ChannelSubscriber(TOPIC_STATE[side], HandState_)
        self._sub.Init(self._on_state, 1)

    def _on_state(self, msg):
        if msg is None or not msg.motor_state:
            return
        deg = [msg.motor_state[i].q for i in range(min(SHARPA_DOF, len(msg.motor_state)))]
        with self._lock:
            self._deg = deg
            self._stamp = time.time()
            self._count += 1

    def get(self):
        """Return (angles_deg ndarray or None, age_seconds, msg_count)."""
        with self._lock:
            if self._deg is None:
                return None, float("inf"), self._count
            return np.array(self._deg, dtype=np.float64), time.time() - self._stamp, self._count


def main():
    ap = argparse.ArgumentParser(description="Read-only Sharpa hand joint angle monitor.")
    ap.add_argument("--side", choices=["left", "right", "both"], default="both")
    ap.add_argument("--unit", choices=["deg", "rad"], default="deg")
    ap.add_argument("--hz", type=float, default=2.0, help="print rate (default: 2)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 = until Ctrl+C")
    ap.add_argument("--dds-domain", type=int, default=0)
    ap.add_argument("--network-interface", default=None, metavar="IFACE",
                    help="NIC for DDS, e.g. eno2")
    ap.add_argument("--save", default=None, metavar="NPZ",
                    help="save timestamps and angles (degrees) to this .npz on exit")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if args.network_interface:
        ChannelFactoryInitialize(args.dds_domain, args.network_interface)
    else:
        ChannelFactoryInitialize(args.dds_domain)

    sides = ["left", "right"] if args.side == "both" else [args.side]
    readers = {s: SharpaStateReader(s) for s in sides}
    log = {s: {"t": [], "q": []} for s in sides}

    scale = math.pi / 180.0 if args.unit == "rad" else 1.0
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print(f"Listening on {[TOPIC_STATE[s] for s in sides]} (unit={args.unit}). Ctrl+C to stop.")

    t0 = time.time()
    period = 1.0 / max(args.hz, 1e-3)
    try:
        while args.duration <= 0 or time.time() - t0 < args.duration:
            for s in sides:
                q, age, n = readers[s].get()
                if q is None:
                    print(f"[{s:5s}] no data yet (bridge on Thor running? correct --network-interface?)")
                    continue
                log[s]["t"].append(time.time())
                log[s]["q"].append(q)
                print(f"[{s:5s}] age={age * 1000:5.0f}ms msgs={n}\n{q * scale}")
            time.sleep(period)
    except KeyboardInterrupt:
        pass

    if args.save:
        out = {}
        for s in sides:
            out[f"{s}_t"] = np.array(log[s]["t"])
            out[f"{s}_q_deg"] = np.array(log[s]["q"])
        np.savez(args.save, **out)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
