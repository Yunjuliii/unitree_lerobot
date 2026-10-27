#!/usr/bin/env python3
"""
Set ``exclude_state: true`` in a GR00T checkpoint's processor_config.json.

Why: trocar_cotrain_rel was trained with --state_dropout_prob 1.0, i.e. the
model never saw the state token.  GR00T's processor only applies that dropout
while training; at inference it feeds the REAL normalized state unless
``exclude_state`` is set (processing_gr00t_n1d7.py: ``if exclude_state or
(state_dropout_prob > 0 and random() < p and self.training)``).  With
``exclude_state: true`` the state tensor given to the model is all zeros -- the
exact token seen in training -- while the raw state is still passed to
decode_action for the relative -> absolute composition.

Run this ONCE on the server host before starting run_gr00t_server.py:

    python patch_groot_processor_exclude_state.py /path/to/checkpoint-XXXXX          # patch (keeps a .orig backup)
    python patch_groot_processor_exclude_state.py /path/to/checkpoint-XXXXX --check  # only print the relevant settings
    python patch_groot_processor_exclude_state.py /path/to/checkpoint-XXXXX --unset  # revert to exclude_state=false
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def summarize(cfg: dict) -> None:
    kw = cfg.get("processor_kwargs", cfg)
    for key in ("exclude_state", "state_dropout_prob", "use_relative_action", "use_percentiles",
                "use_mean_std", "max_action_horizon", "max_state_dim", "max_action_dim"):
        print(f"  {key:22s} = {kw.get(key)!r}")
    for emb, mc in kw.get("modality_configs", {}).items():
        print(f"  embodiment {emb!r}:")
        for mod in ("video", "state", "action", "language"):
            m = mc.get(mod, {})
            extra = ""
            if mod == "action":
                extra = f" horizon={len(m.get('delta_indices', []))} configs={[(a.get('rep'), a.get('type'), a.get('format')) for a in (m.get('action_configs') or [])]}"
            print(f"    {mod:9s} keys={m.get('modality_keys')}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="checkpoint directory containing processor_config.json")
    ap.add_argument("--check", action="store_true", help="print the settings, change nothing")
    ap.add_argument("--unset", action="store_true", help="set exclude_state back to false")
    args = ap.parse_args()

    # Same lookup as Gr00tPolicy: <ckpt>/processor_config.json, else <ckpt>/processor/processor_config.json
    root = os.path.expanduser(args.checkpoint)
    candidates = [os.path.join(root, "processor_config.json"), os.path.join(root, "processor", "processor_config.json")]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        print(f"ERROR: no processor config found; looked for {candidates}", file=sys.stderr)
        return 1
    with open(path) as f:
        cfg = json.load(f)
    kw = cfg.get("processor_kwargs")
    if not isinstance(kw, dict):
        print(f"ERROR: {path} has no 'processor_kwargs' dict", file=sys.stderr)
        return 1

    print(f"{path}:")
    summarize(cfg)
    if not kw.get("use_relative_action", False):
        print("\nWARNING: use_relative_action is false -- this processor would neither compose relative->absolute nor "
              "un-normalise the eef groups with the relative_action statistics; the eef rows it returns are unusable. "
              "The trainer forces use_relative_action=True; this checkpoint's processor config is not the one it trained with.",
              file=sys.stderr)
    if args.check:
        return 0

    target = not args.unset
    if bool(kw.get("exclude_state", False)) == target:
        print(f"\nexclude_state already {target}; nothing to do.")
        return 0

    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"\nbackup written: {backup}")
    kw["exclude_state"] = target
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)
    print(f"exclude_state set to {target} in {path}. Restart run_gr00t_server.py to pick it up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
