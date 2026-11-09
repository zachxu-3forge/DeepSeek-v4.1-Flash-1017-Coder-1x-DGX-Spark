#!/usr/bin/env python3
"""Translate the omlx quantization map into TensorFold's affine metadata.

The target checkpoint describes quantization under
`omlx_deepseek_v41.quantized_modules`. TensorFold's affine reader expects a
`quantization` object with a global `bits`/`group_size` and per-module overrides
keyed by canonical module names. This script emits that object and can write a
translated copy of config.json.

Examples:
  python3 scripts/mlx/translate_config.py MODEL_DIR
  python3 scripts/mlx/translate_config.py MODEL_DIR --out quantization.json
  python3 scripts/mlx/translate_config.py MODEL_DIR --write-config config.translated.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

QUANT_ROOT = "omlx_deepseek_v41"
PREFIXES = ("model.language_model.", "language_model.", "text_model.", "model.")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical(key: str) -> str:
    k = key
    if k.endswith(".weight"):
        k = k[: -len(".weight")]
    for prefix in PREFIXES:
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="checkpoint directory")
    ap.add_argument("--out", help="write the canonical quantization object here")
    ap.add_argument("--write-config", help="write a translated config.json copy here")
    ap.add_argument("--check", action="store_true", help="verify weight/scales/biases exist")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a quantized module is incomplete")
    args = ap.parse_args(argv)

    model_dir = os.path.abspath(args.model_dir)
    config_path = os.path.join(model_dir, "config.json")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(config_path):
        print(f"missing {config_path}", file=sys.stderr)
        return 2

    config = load_json(config_path)
    source = config.get(QUANT_ROOT, {}).get("quantized_modules")
    if not isinstance(source, dict) or not source:
        print(f"missing {QUANT_ROOT}.quantized_modules", file=sys.stderr)
        return 2

    overrides: dict[str, dict | bool] = {}
    non_affine: dict[str, dict] = {}
    alias_collisions: list[str] = []
    seen: dict[str, str] = {}
    for name, spec in source.items():
        if not isinstance(spec, dict):
            print(f"{name}: not an object; refused", file=sys.stderr)
            return 2
        key = canonical(name)
        if key in seen and seen[key] != name:
            alias_collisions.append(f"{name} -> {key} (already {seen[key]})")
        seen[key] = name
        mode = spec.get("mode", "affine")
        if mode != "affine":
            non_affine[key] = {
                "mode": mode,
                "bits": spec.get("bits"),
                "group_size": spec.get("group_size"),
            }
            overrides[key] = False
            continue
        out = {
            "bits": spec.get("bits"),
            "group_size": spec.get("group_size", 64),
            "mode": "affine",
        }
        if "quantize_input" in spec:
            out["quantize_input"] = bool(spec["quantize_input"])
        if out["bits"] not in (2, 3, 4, 5, 6, 8):
            print(f"{name}: unsupported bits {out['bits']!r}", file=sys.stderr)
            return 2
        if out["group_size"] not in (32, 64, 128):
            print(f"{name}: unsupported group {out['group_size']!r}", file=sys.stderr)
            return 2
        overrides[key] = out

    if alias_collisions:
        print("alias collisions:", file=sys.stderr)
        for line in alias_collisions:
            print("  " + line, file=sys.stderr)
        return 2

    quant = {
        "quant_method": "affine",
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "dense_by_default": True,
        "non_affine_modules": non_affine,
        **overrides,
    }

    affine_count = len(overrides) - len(non_affine)
    print(f"model: {model_dir}")
    print(f"quantized modules: {len(source)} (affine {affine_count}, non-affine {len(non_affine)})")
    print(f"canonical sample: {next(iter(overrides))}")
    if non_affine:
        sample = ", ".join(f"{k}={v['mode']}" for k, v in list(non_affine.items())[:5])
        print(f"non-affine sample: {sample}")
    if args.check:
        index = load_json(index_path)
        weight_map = index.get("weight_map", {})
        missing = []
        for name, spec in source.items():
            if spec.get("mode", "affine") == "affine":
                suffixes = (".weight", ".scales", ".biases")
            else:
                suffixes = (".weight", ".scales")
            for suffix in suffixes:
                if name + suffix not in weight_map:
                    missing.append(name + suffix)
        if missing:
            print(f"missing affine pieces: {len(missing)}")
            for m in missing[:20]:
                print("  " + m)
        else:
            print("affine pieces: complete")
        if missing and args.strict:
            return 1

    text = json.dumps(quant, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)

    if args.write_config:
        translated = dict(config)
        translated["quantization"] = quant
        translated["quantization_config"] = quant
        with open(args.write_config, "w", encoding="utf-8") as f:
            json.dump(translated, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.write_config}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
