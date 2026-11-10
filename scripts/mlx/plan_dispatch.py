#!/usr/bin/env python3
"""Independent expected dispatch counts for the affine/mixed checkpoint.

Run before the engine so the boot log's `dispatch:` line can be checked against a
second implementation. Reads only config.json and the index; no weights.

Example:
  python3 scripts/mlx/plan_dispatch.py "$MODEL_DIR" --json out/plan_dispatch.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

PREFIXES = ("model.language_model.", "language_model.", "text_model.", "model.")


def canonical(key: str) -> str:
    k = key
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    for p in PREFIXES:
        if k.startswith(p):
            k = k[len(p):]
    return k


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    model = os.path.abspath(args.model_dir)
    with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    modules = config.get("omlx_deepseek_v41", {}).get("quantized_modules", {})
    if not modules:
        print("missing omlx_deepseek_v41.quantized_modules", file=sys.stderr)
        return 2

    counts: Counter = Counter()
    errors: list[str] = []
    for name, spec in modules.items():
        key = canonical(name)
        mode = spec.get("mode", "affine")
        expert = ".ffn.experts." in key
        bits = spec.get("bits")
        group = spec.get("group_size")
        if mode == "mxfp8":
            if expert:
                errors.append(f"{key}: MXFP8 on an expert")
            elif key.endswith(".main_proj"):
                counts["dspark_main_proj"] += 1
            else:
                counts["mxfp8"] += 1
        elif mode == "mxfp4":
            if not expert:
                errors.append(f"{key}: MXFP4 on a linear")
            counts["mxfp4"] += 1
        elif mode == "affine":
            if group != 64:
                errors.append(f"{key}: affine group {group}, expected 64")
            if expert:
                if bits in (3, 4):
                    counts["affine_native_expert"] += 1
                elif bits == 2:
                    counts["affine_n_expert"] += 1
                else:
                    errors.append(f"{key}: unsupported expert bits {bits}")
            elif ".engram." in key:
                counts["affine_engram"] += 1
            else:
                if bits in (2, 3, 5, 6, 8):
                    counts["affine8_linear"] += 1
                else:
                    errors.append(f"{key}: unsupported linear bits {bits}")
        else:
            errors.append(f"{key}: unknown mode {mode!r}")

    result = {
        "model": model,
        "quantized_modules": len(modules),
        "counts": {
            "affine8_linear": counts["affine8_linear"],
            "affine_n_expert": counts["affine_n_expert"],
            "affine_native_expert": counts["affine_native_expert"],
            "affine_engram": counts["affine_engram"],
            "mxfp8": counts["mxfp8"],
            "mxfp4": counts["mxfp4"],
            "dspark_main_proj": counts["dspark_main_proj"],
        },
        "errors": errors,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
