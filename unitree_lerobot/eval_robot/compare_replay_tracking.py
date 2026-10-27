"""Compare arm tracking error of two replay_h2.py runs (gravity comp OFF vs ON).

Usage:
    python compare_replay_tracking.py <gc_off.npz> <gc_on.npz>
"""

import sys

import numpy as np

LABELS = [
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow", "L_wr_roll", "L_wr_pitch", "L_wr_yaw",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow", "R_wr_roll", "R_wr_pitch", "R_wr_yaw",
]


def stats(path):
    d = np.load(path)
    err = d["q_meas"] - d["q_target"]
    return bool(d["gravity_comp"]), np.abs(err).mean(axis=0), np.abs(err).max(axis=0), err.mean(axis=0)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    a, b = sys.argv[1], sys.argv[2]
    gc_a, mean_a, max_a, bias_a = stats(a)
    gc_b, mean_b, max_b, bias_b = stats(b)
    print(f"A = {a}  (gravity_comp={gc_a})")
    print(f"B = {b}  (gravity_comp={gc_b})")
    print(f"\n{'joint':<11}{'mean|e| A':>11}{'mean|e| B':>11}{'ratio B/A':>11}{'bias A':>9}{'bias B':>9}{'max A':>8}{'max B':>8}")
    for i, name in enumerate(LABELS):
        ratio = mean_b[i] / mean_a[i] if mean_a[i] > 1e-9 else float("nan")
        print(f"{name:<11}{mean_a[i]:11.4f}{mean_b[i]:11.4f}{ratio:11.2f}{bias_a[i]:+9.4f}{bias_b[i]:+9.4f}{max_a[i]:8.4f}{max_b[i]:8.4f}")
    print(f"{'ALL':<11}{mean_a.mean():11.4f}{mean_b.mean():11.4f}{mean_b.mean()/mean_a.mean():11.2f}")


if __name__ == "__main__":
    main()
