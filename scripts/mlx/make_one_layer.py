#!/usr/bin/env python3
"""Build a one-layer fixture from the checkpoint for `TF_DSV41_LAYERS=<layer>-<layer>`.

Copies the selected layer, the embedding, final norm, and head into a single
safetensors file, copies the tokenizer/chat template, and translates the config.
The result is small enough to load quickly for the Stage 1 one-layer tests.

Examples:
  python3 scripts/mlx/make_one_layer.py "$MODEL_DIR" /models/mlx-one-layer --layer 0
  python3 scripts/mlx/make_one_layer.py "$MODEL_DIR" /models/mlx-one-layer --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

TOP = ("language_model.embed.weight", "language_model.norm.weight", "language_model.head.weight")
COPY = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")
CHUNK = 8 << 20


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        blob = f.read(n)
    return json.loads(blob), n


def select_names(weight_map: dict, layer: int) -> list[str]:
    prefix = f"language_model.layers.{layer}."
    names = [n for n in weight_map if n.startswith(prefix)]
    names += [n for n in TOP if n in weight_map]
    return sorted(dict.fromkeys(names))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="list selected tensors and exit")
    args = ap.parse_args(argv)

    model = os.path.abspath(args.model_dir)
    out = os.path.abspath(args.out_dir)
    index = load_json(os.path.join(model, "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    names = select_names(weight_map, args.layer)
    if not names:
        print(f"no tensors for layer {args.layer}", file=sys.stderr)
        return 2

    if args.list:
        for n in names:
            print(n)
        print(f"tensors: {len(names)}")
        return 0

    os.makedirs(out, exist_ok=True)
    headers: dict[str, tuple[dict, int]] = {}
    entries = []
    total = 0
    for name in names:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = read_header(os.path.join(model, shard))
        header, base = headers[shard]
        info = header[name]
        begin, end = info["data_offsets"]
        entries.append((name, info["dtype"], info["shape"], shard, base + begin, end - begin))
        total += end - begin

    # safetensors header, padded to 8 bytes
    head = {"__metadata__": {"format": "pt"}}
    at = 0
    for name, dtype, shape, _shard, _start, length in entries:
        head[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + length]}
        at += length
    blob = json.dumps(head, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(blob) % 8) % 8
    if pad:
        blob += b" " * pad

    shard_name = "model.safetensors"
    with open(os.path.join(out, shard_name), "wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        for name, _dtype, _shape, shard, start, length in entries:
            src_path = os.path.join(model, shard)
            with open(src_path, "rb") as src:
                src.seek(start)
                left = length
                while left:
                    chunk = src.read(min(CHUNK, left))
                    if not chunk:
                        raise IOError(f"{name}: short read")
                    dst.write(chunk)
                    left -= len(chunk)

    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": {n: shard_name for n in names}}, f, indent=2)

    shutil.copy2(os.path.join(model, "config.json"), os.path.join(out, "config.json"))
    for name in COPY:
        src = os.path.join(model, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, name))

    translate = os.path.join(os.path.dirname(__file__), "translate_config.py")
    translated = os.path.join(out, "config.translated.json")
    subprocess.check_call([sys.executable, translate, out, "--write-config", translated], stdout=subprocess.DEVNULL)
    os.replace(translated, os.path.join(out, "config.json"))

    print(f"wrote {out}: {len(names)} tensors, {total / (1 << 30):.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
