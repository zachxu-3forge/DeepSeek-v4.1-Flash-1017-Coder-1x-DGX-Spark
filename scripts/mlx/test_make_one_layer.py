#!/usr/bin/env python3
"""Regression test for `make_one_layer.py`: the fixture's bytes must equal the source shard's.

Builds a tiny shard whose header is longer than the first tensor, runs the extractor, and
compares every copied tensor byte-for-byte. This catches offsets taken relative to the data
section instead of the file.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile


def write_shard(path: str, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {"__metadata__": {"format": "pt"}}
    at = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(data)]}
        at += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _dtype, _shape, data in tensors.values():
            f.write(data)


def read_shard(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_bytes(path: str, name: str) -> bytes:
    header, base = read_shard(path)
    begin, end = header[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + begin)
        return f.read(end - begin)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "model")
        out = os.path.join(tmp, "one-layer")
        os.makedirs(model)
        tensors = {
            "language_model.layers.0.attn.wq_a.weight": (
                "U32",
                [2, 2],
                bytes.fromhex("01020304" "11121314"),
            ),
            "language_model.layers.0.attn.wq_a.scales": ("BF16", [2, 1], bytes.fromhex("a1a2a3a4")),
            "language_model.layers.0.attn.wq_a.biases": ("BF16", [2, 1], bytes.fromhex("b1b2b3b4")),
        }
        shard = "model-00001-of-00021.safetensors"
        write_shard(os.path.join(model, shard), tensors)
        index = {"weight_map": {name: shard for name in tensors}}
        with open(os.path.join(model, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f)
        config = {
            "model_type": "deepseek_v41",
            "omlx_deepseek_v41": {
                "quantized_modules": {
                    "language_model.layers.0.attn.wq_a": {
                        "bits": 8,
                        "group_size": 64,
                        "mode": "affine",
                    }
                }
            },
        }
        with open(os.path.join(model, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

        subprocess.check_call(
            [sys.executable, os.path.join(here, "make_one_layer.py"), model, out, "--layer", "0"],
            stdout=subprocess.DEVNULL,
        )
        fixture = os.path.join(out, "model.safetensors")
        for name, (_dtype, _shape, want) in tensors.items():
            got = tensor_bytes(fixture, name)
            assert got == want, (name, got.hex(), want.hex())
        print("one-layer fixture: bytes match the source shard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
