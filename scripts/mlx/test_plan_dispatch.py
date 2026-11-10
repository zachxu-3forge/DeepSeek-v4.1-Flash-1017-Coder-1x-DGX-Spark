#!/usr/bin/env python3
"""Checks `plan_dispatch.py`'s affine-plan buckets.

Engram tables and DSpark's `main_proj` are not affine-plan modules; counting them as
ordinary affine8/MXFP8 linears made the engine guard look short by 2 and 1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    modules = {
        "language_model.layers.0.attn.wq_a": {"bits": 8, "group_size": 64, "mode": "affine"},
        "language_model.layers.0.ffn.experts.w1": {"bits": 2, "group_size": 64, "mode": "affine"},
        "language_model.layers.1.engram.wkv": {"bits": 6, "group_size": 64, "mode": "affine"},
        "language_model.mtp.0.attn.wkv": {"bits": 8, "group_size": None, "mode": "mxfp8"},
        "language_model.mtp.0.ffn.experts.w1": {"bits": 4, "group_size": None, "mode": "mxfp4"},
        "language_model.mtp.0.main_proj": {"bits": 8, "group_size": None, "mode": "mxfp8"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"omlx_deepseek_v41": {"quantized_modules": modules}}, f)
        out = os.path.join(tmp, "plan.json")
        subprocess.check_call(
            [sys.executable, os.path.join(here, "plan_dispatch.py"), tmp, "--json", out],
            stdout=subprocess.DEVNULL,
        )
        counts = json.load(open(out, encoding="utf-8"))["counts"]
        want = {
            "affine8_linear": 1,
            "affine_n_expert": 1,
            "affine_engram": 1,
            "mxfp8": 1,
            "mxfp4": 1,
            "dspark_main_proj": 1,
        }
        for key, value in want.items():
            assert counts[key] == value, (key, counts)
        print("plan dispatch: affine-plan buckets match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
