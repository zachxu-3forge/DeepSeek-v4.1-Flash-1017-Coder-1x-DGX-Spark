#!/usr/bin/env python3
"""Inspect the MLX checkpoint without loading weights.

Reads config.json, model.safetensors.index.json, and (with --headers) the safetensors
headers. Reports the quantization map, shard sizes, stacked-expert layout, Engram layout,
and packed-affine shape checks.

Example:
  python3 scripts/mlx/inspect_checkpoint.py /models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-MLX --headers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter

QUANT_ROOT = "omlx_deepseek_v41"

DEFAULT_PATTERNS = [
    r"\.ffn\.experts\.(w1|w2|w3)\.(weight|scales|biases)$",
    r"\.engram\.(embed|wkv)\.(weight|scales|biases)$",
    r"\.attn\.wo_a\.weight$",
]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if n == 0 or n > (256 << 20):
            raise ValueError(f"{path}: implausible header length {n}")
        blob = f.read(n)
    if len(blob) != n:
        raise ValueError(f"{path}: truncated header")
    return json.loads(blob)


def module_of(name: str) -> str:
    for suffix in (".weight", ".scales", ".biases"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def bits_summary(modules: dict) -> tuple[Counter, Counter, Counter]:
    bits = Counter()
    groups = Counter()
    modes = Counter()
    for spec in modules.values():
        if isinstance(spec, dict):
            bits[spec.get("bits")] += 1
            groups[spec.get("group_size")] += 1
            modes[spec.get("mode")] += 1
        else:
            bits["<non-object>"] += 1
    return bits, groups, modes


def human_gib(n: int) -> str:
    return f"{n / (1 << 30):.1f} GiB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="checkpoint directory")
    ap.add_argument("--headers", action="store_true", help="read safetensors headers")
    ap.add_argument("--pattern", action="append", default=[], help="tensor-name regex (repeatable)")
    ap.add_argument("--limit", type=int, default=20, help="max matching tensors printed per pattern")
    ap.add_argument("--json", dest="json_out", help="write the report as JSON to this path ('-' for stdout)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on missing quantized tensors")
    args = ap.parse_args(argv)

    model_dir = os.path.abspath(args.model_dir)
    config_path = os.path.join(model_dir, "config.json")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(config_path):
        print(f"missing {config_path}", file=sys.stderr)
        return 2
    if not os.path.isfile(index_path):
        print(f"missing {index_path}", file=sys.stderr)
        return 2

    config = load_json(config_path)
    text = config.get("text_config", config)
    modules = config.get(QUANT_ROOT, {}).get("quantized_modules", {})
    index = load_json(index_path)
    weight_map = index.get("weight_map", {})
    shard_bytes = index.get("metadata", {}).get("total_size", 0)
    shard_counts = Counter(weight_map.values())
    if not shard_bytes:
        shard_bytes = sum(
            os.path.getsize(os.path.join(model_dir, s)) for s in shard_counts
        )

    bits, groups, modes = bits_summary(modules)
    stacked = sorted(
        n for n in weight_map
        if ".ffn.experts." in n and not re.search(r"\.experts\.\d+\.", n)
    )
    per_expert = sorted(
        n for n in weight_map
        if ".ffn.experts." in n and re.search(r"\.experts\.\d+\.", n)
    )
    engram = sorted(n for n in weight_map if ".engram." in n)

    report: dict = {
        "model_dir": model_dir,
        "model_type": config.get("model_type"),
        "text": {
            k: text.get(k)
            for k in ("num_hidden_layers", "hidden_size", "n_routed_experts",
                      "num_experts_per_tok", "moe_intermediate_size", "vocab_size")
        },
        "quantized_modules": len(modules),
        "bits": {str(k): v for k, v in bits.items()},
        "groups": {str(k): v for k, v in groups.items()},
        "modes": {str(k): v for k, v in modes.items()},
        "shards": len(shard_counts),
        "tensors": len(weight_map),
        "shard_bytes": shard_bytes,
        "stacked_expert_tensors": len(stacked),
        "per_expert_tensors": len(per_expert),
        "engram_tensors": len(engram),
        "headers": {},
    }

    print(f"model: {model_dir}")
    print(f"model_type: {config.get('model_type')}")
    print("text: " + ", ".join(f"{k}={v}" for k, v in report["text"].items()))
    print(f"quantized modules: {len(modules)}")
    print("bits: " + ", ".join(f"{k}={v}" for k, v in sorted(bits.items(), key=lambda x: str(x[0]))))
    print("groups: " + ", ".join(f"{k}={v}" for k, v in sorted(groups.items(), key=lambda x: str(x[0]))))
    print("modes: " + ", ".join(f"{k}={v}" for k, v in sorted(modes.items(), key=lambda x: str(x[0]))))
    print(f"index: {len(shard_counts)} shards, {len(weight_map)} tensors, {human_gib(shard_bytes)}")
    print(f"stacked expert tensors: {len(stacked)}; per-expert tensors: {len(per_expert)}")
    print(f"engram tensors: {len(engram)}")
    if stacked:
        print("stacked samples: " + ", ".join(stacked[:5]))
    if engram:
        print("engram samples: " + ", ".join(engram[:5]))

    missing = []
    for mod, spec in modules.items():
        mode = spec.get("mode", "affine") if isinstance(spec, dict) else "affine"
        suffixes = (".weight", ".scales", ".biases") if mode == "affine" else (".weight", ".scales")
        for suffix in suffixes:
            if mod + suffix not in weight_map:
                missing.append(mod + suffix)
    if missing:
        print(f"missing quantized pieces: {len(missing)} (first: {missing[:5]})")
        report["missing_quantized_pieces"] = missing
    else:
        print("quantized pieces: every module has its weight and metadata")

    if args.headers:
        patterns = args.pattern or DEFAULT_PATTERNS
        compiled = [re.compile(p) for p in patterns]
        seen = set()
        counts = [0] * len(compiled)
        for shard in sorted(shard_counts):
            path = os.path.join(model_dir, shard)
            header = read_safetensors_header(path)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                matched = [i for i, c in enumerate(compiled) if c.search(name)]
                if not matched or name in seen:
                    continue
                if all(counts[i] >= args.limit for i in matched):
                    continue
                seen.add(name)
                for i in matched:
                    counts[i] += 1
                entry = {"dtype": info.get("dtype"), "shape": info.get("shape"), "shard": shard}
                report["headers"][name] = entry
                mod = module_of(name)
                spec = modules.get(mod)
                shape = info.get("shape") or []
                check = ""
                if spec and spec.get("mode", "affine") == "affine" and len(shape) >= 2:
                    bits = spec.get("bits")
                    group = spec.get("group_size")
                    last = shape[-1]
                    if name.endswith(".weight") and bits:
                        if last * 32 % bits == 0:
                            implied = last * 32 // bits
                            check = f" bits={bits} packed_last={last} implied_k={implied}"
                            if group and implied % group:
                                check += " (not group-aligned)"
                        else:
                            check = f" bits={bits} packed_last={last} (not 32-bit aligned)"
                    elif name.endswith((".scales", ".biases")):
                        check = f" group={group} implied_k={last * group if group else '?'}"
                print(f"{name}: {entry['dtype']} {shape}{check}")
        print(f"matched tensors: {len(seen)}")

    if args.json_out:
        text_out = json.dumps(report, indent=2, sort_keys=True)
        if args.json_out == "-":
            print(text_out)
        else:
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(text_out + "\n")
            print(f"wrote {args.json_out}")

    return 1 if (args.strict and missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
