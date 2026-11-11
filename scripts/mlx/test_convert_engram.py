#!/usr/bin/env python3
"""Unit checks for the Engram converter: E4M3/E8M0 math, native rows, and a tiny shard."""

from __future__ import annotations

import json
import os
import struct
import tempfile

import numpy as np

import convert_engram as c
import test_engram as te


def write_tiny_model(root: str) -> None:
    """A release with one 8-row affine-4/group-32 Engram layer and a matching index."""
    rng = np.random.default_rng(11)
    words = rng.integers(0, 2**32, size=(8, 32), dtype=np.uint32).astype("<u4")
    s16 = ((rng.standard_normal((8, 8)).astype(np.float32) * 0.05).view(np.uint32) >> 16).astype("<u2")
    b16 = ((rng.standard_normal((8, 8)).astype(np.float32) * 0.05).view(np.uint32) >> 16).astype("<u2")
    blobs = [words.tobytes(), s16.tobytes(), b16.tobytes()]
    names = [
        "language_model.layers.1.engram.embed.weight",
        "language_model.layers.1.engram.embed.scales",
        "language_model.layers.1.engram.embed.biases",
    ]
    offsets = []
    at = 0
    for b in blobs:
        offsets.append([at, at + len(b)])
        at += len(b)
    header = {}
    for name, blob, (lo, hi) in zip(names, blobs, offsets):
        header[name] = {
            "dtype": "U32" if name.endswith("weight") else "BF16",
            "shape": [8, 32] if name.endswith("weight") else [8, 8],
            "data_offsets": [lo, hi],
        }
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "model-00001-of-00001.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for b in blobs:
            f.write(b)
    config = {
        "text_config": {"engram_layer_ids": [1], "engram_num_embeddings": [8]},
        "omlx_deepseek_v41": {"quantized_modules": {}},
    }
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}
    with open(os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)


def main() -> int:
    vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 448.0, -1.0, -448.0, 2.0**-9], dtype=np.float32)
    bits = c.e4m3_bits(vals)
    back = c.e4m3_to_f32(bits)
    err = np.abs(back - vals)
    assert err.max() <= 0.5, (vals, bits, back, err)
    assert c.e4m3_to_f32(np.array([0x7E], dtype=np.uint8))[0] == 448.0

    rng = np.random.default_rng(7)
    rows = (rng.standard_normal((64, 256)) * 3).astype(np.float32)
    codes, scales = c.quantize_row(rows)
    recovered = c.dequantize_fp8(codes, scales)
    rms = float(np.sqrt(np.mean((recovered - rows) ** 2)))
    assert rms < 0.15, rms
    assert scales.min() >= 1 and scales.max() <= 254
    print(f"e4m3/e8m0: ok (roundtrip rms {rms:.4f})")

    words = rng.integers(0, 2**32, size=(64, 32), dtype=np.uint32)
    s16 = ((rng.standard_normal((64, 8)).astype(np.float32) * 0.05).view(np.uint32) >> 16).astype(np.uint16)
    b16 = ((rng.standard_normal((64, 8)).astype(np.float32) * 0.05).view(np.uint32) >> 16).astype(np.uint16)
    native = c.native_chunk(words.tobytes(), s16.tobytes(), b16.tobytes())
    assert native.shape == (64, c.NATIVE_ROW_BYTES)
    expected = c.dequantize(words, s16, b16)
    assert np.array_equal(c.dequantize_native(native), expected)
    assert np.array_equal(native[:, :128], words.view(np.uint8).reshape(64, 128))
    assert np.array_equal(native[:, 128:144], s16.view(np.uint8).reshape(64, 16))
    assert np.array_equal(native[:, 144:160], b16.view(np.uint8).reshape(64, 16))
    print("native rows: ok (160-byte interleave)")

    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "model")
        native_dir = os.path.join(tmp, "native")
        fp8_dir = os.path.join(tmp, "fp8")
        write_tiny_model(model)
        assert c.main([model, native_dir, "--verify"]) == 0
        path = os.path.join(native_dir, "engram-l1-r0of1.bin")
        with open(path, "rb") as f:
            header = struct.unpack("<6Q", f.read(48))
        assert header == (c.MAGIC, 1, 0, 8, 8, c.NATIVE_ROW_BYTES), header
        assert os.path.getsize(path) == c.HEADER_BYTES + 8 * c.NATIVE_ROW_BYTES
        assert te.main([model, "--shard-dir", native_dir]) == 0
        assert c.main([model, fp8_dir, "--format", "fp8", "--verify"]) == 0
        assert os.path.getsize(os.path.join(fp8_dir, "engram-l1-r0of1.bin")) == c.HEADER_BYTES + 8 * c.FP8_ROW_BYTES
    print("tiny native/fp8 shard: ok (header, rows, test_engram --shard-dir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
