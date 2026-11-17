#!/usr/bin/env python3
"""Convert the checkpoint's MXFP8/MXFP4 DSpark modules to MLX affine.

The MLX checkpoint keeps the three DSpark (`mtp.0..2`) blocks in the release's
native formats: attention/shared/main projections are E4M3 bytes with one E8M0
scale per 32 inputs, the routed experts are E2M1 nibbles with the same scales.
The engine's affine path serves group-64 affine tiles; this script requantizes
the DSpark modules into that form so one kernel family serves the whole model.

Targets: linears (including `main_proj`) -> affine-8/group 64; routed experts ->
affine-3/group 64 (the affine-8 kernel family serves 2/3/5/6/8-bit codes, not 4).
The converted tensors are written to one new safetensors shard and the model's
index/config are updated to match. The source shards are not modified; the output
directory hardlinks them.

Example:
  python3 scripts/mlx/convert_dspark_affine.py "$MODEL_DIR" "$OUT_DIR"
  python3 scripts/mlx/convert_dspark_affine.py "$MODEL_DIR" "$OUT_DIR" --verify
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

HEADER = 8
GIB = 1 << 30
SRC_GROUP = 32  # E8M0 scale bytes cover 32 inputs
DST_GROUP = 64
ROWS_BLOCK = 512

# OCP E4M3 (float8_e4m3fn); 0x7F/0xFF are NaN and do not occur in the checkpoint.
_E4M3 = np.zeros(256, dtype=np.float32)
for _c in range(256):
    _e = (_c >> 3) & 15
    _m = np.float32(_c & 7)
    if _e == 15 and (_c & 7) == 7:
        _v = np.float32("nan")
    elif _e == 0:
        _v = _m * np.float32(2.0**-9)
    else:
        _v = (8 + _m) * np.float32(2.0 ** (_e - 10))
    _E4M3[_c] = -_v if (_c & 0x80) else _v

_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6] + [-0, -0.5, -1, -1.5, -2, -3, -4, -6],
                 dtype=np.float32)


def e8m0(scale: np.ndarray) -> np.ndarray:
    """E8M0 byte -> the float it scales by (2^(b-127))."""
    return np.exp2(scale.astype(np.int32) - 127).astype(np.float32)


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round f32 to bf16 (nearest even) and return the uint16 bits."""
    u = values.astype(np.float32).view(np.uint32).copy()
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    out = rounded.astype(np.uint16)
    out[nan] = ((u[nan] >> 16) & 0x8000) | 0x7FC0
    return out


def decode_mxfp8(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """U32 words [rows, k/4] plus E8M0 [rows, k/32] -> f32 [rows, k]."""
    codes = weight.view(np.uint8).reshape(weight.shape[0], -1)
    values = _E4M3[codes]
    k = codes.shape[1]
    out = values * np.repeat(e8m0(scales), SRC_GROUP, axis=1)[:, :k]
    out[~np.isfinite(out)] = 0.0  # 0x7F/0xFF are NaN codes; the checkpoint has none
    return out


def decode_mxfp4(weight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """U32 words [rows, k/8] plus E8M0 [rows, k/32] -> f32 [rows, k]."""
    raw = weight.view(np.uint8).reshape(weight.shape[0], -1)
    codes = np.empty((raw.shape[0], raw.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = raw & 0x0F
    codes[:, 1::2] = raw >> 4
    values = _E2M1[codes]
    k = codes.shape[1]
    out = values * np.repeat(e8m0(scales), SRC_GROUP, axis=1)[:, :k]
    out[~np.isfinite(out)] = 0.0
    return out


def quantize_affine(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[rows, k] f32 -> affine codes (uint8), bf16 scale/bias bits [rows, k/64]."""
    rows, k = values.shape
    groups = k // DST_GROUP
    v = values.reshape(rows, groups, DST_GROUP).astype(np.float32)
    lo = v.min(axis=2)
    hi = v.max(axis=2)
    max_code = (1 << bits) - 1
    scale = (hi - lo) / np.float32(max_code)
    scale = np.where(scale > 0, scale, np.float32(1.0))
    bias = lo
    q = np.rint((v - bias[:, :, None]) / scale[:, :, None])
    q = np.clip(q, 0, max_code).astype(np.uint8).reshape(rows, k)
    return q, bf16_round(scale).reshape(rows, groups), bf16_round(bias).reshape(rows, groups)


def pack_codes(q: np.ndarray, bits: int) -> np.ndarray:
    """uint8 codes [rows, k] -> U32 words [rows, k*bits/32] (little-endian, LSB first)."""
    rows, k = q.shape
    if bits == 8:
        body = np.ascontiguousarray(q.astype(np.uint8))
        return body.view(np.uint32).reshape(rows, k // 4)
    if k * bits % 32 != 0:
        raise ValueError(f"{bits}-bit codes of k={k} do not fill whole words")
    # bit j of a code is bit (i * bits + j) of the row's little-endian bit stream
    bits_arr = ((q.astype(np.uint16)[:, :, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    flat = bits_arr.reshape(rows, k * bits)
    packed = np.packbits(flat, axis=1, bitorder="little")
    return packed.view(np.uint32).reshape(rows, k * bits // 32)


def unpack_codes(words: np.ndarray, bits: int, k: int) -> np.ndarray:
    """U32 words [rows, k*bits/32] -> uint8 codes [rows, k]."""
    raw = words.view(np.uint8).reshape(words.shape[0], -1)
    flat = np.unpackbits(raw, axis=1, bitorder="little")[:, : k * bits]
    shifts = (1 << np.arange(bits, dtype=np.uint16)).astype(np.uint16)
    return (flat.reshape(-1, k, bits) * shifts).sum(axis=2).astype(np.uint8)


def safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(model: str, index: dict, name: str) -> np.ndarray:
    shard = index["weight_map"][name]
    header, base = safetensors_header(os.path.join(model, shard))
    info = header[name]
    off = base + info["data_offsets"][0]
    nbytes = info["data_offsets"][1] - info["data_offsets"][0]
    dtype = {"U32": "<u4", "U8": "u1", "BF16": "<u2", "F32": "<f4"}.get(info["dtype"])
    if dtype is None:
        raise ValueError(f"{name}: unexpected dtype {info['dtype']}")
    with open(os.path.join(model, shard), "rb") as f:
        f.seek(off)
        raw = f.read(nbytes)
    return np.frombuffer(raw, dtype=dtype).reshape(info["shape"]).copy()


def modules(config: dict) -> dict[str, dict]:
    """Every converted module: name (without language_model.) -> spec."""
    out = {}
    for key, value in (config.get("omlx_deepseek_v41") or {}).get("quantized_modules", {}).items():
        if ".mtp." not in key or not isinstance(value, dict):
            continue
        mode = value.get("mode")
        if mode not in ("mxfp8", "mxfp4"):
            continue
        out[key.replace("language_model.", "", 1)] = value
    return out


def convert_module(model: str, index: dict, name: str, spec: dict, dst_bits: int) -> dict[str, tuple[str, tuple, bytes]]:
    prefix = "language_model." + name
    weight = read_tensor(model, index, prefix + ".weight")
    scales = read_tensor(model, index, prefix + ".scales")
    src_bits = int(spec["bits"])
    leading = weight.shape[:-1]
    if src_bits == 8:
        k = weight.shape[-1] * 4
        flat = weight.reshape(-1, weight.shape[-1])
        sflat = scales.reshape(-1, scales.shape[-1])
        decode = decode_mxfp8
    elif src_bits == 4:
        k = weight.shape[-1] * 8
        flat = weight.reshape(-1, weight.shape[-1])
        sflat = scales.reshape(-1, scales.shape[-1])
        decode = decode_mxfp4
    else:
        raise ValueError(f"{name}: unexpected source bits {src_bits}")
    rows = flat.shape[0]
    out_w = np.empty((rows, k * dst_bits // 32), dtype=np.uint32)
    groups = k // DST_GROUP
    out_s = np.empty((rows, groups), dtype=np.uint16)
    out_b = np.empty((rows, groups), dtype=np.uint16)
    for r0 in range(0, rows, ROWS_BLOCK):
        r1 = min(rows, r0 + ROWS_BLOCK)
        values = decode(flat[r0:r1], sflat[r0:r1])
        q, s, b = quantize_affine(values, dst_bits)
        out_w[r0:r1] = pack_codes(q, dst_bits)
        out_s[r0:r1] = s
        out_b[r0:r1] = b
    del flat, sflat, weight, scales
    wshape = leading + (k * dst_bits // 32,)
    sshape = leading + (groups,)
    return {
        prefix + ".weight": ("U32", wshape, out_w.tobytes()),
        prefix + ".scales": ("BF16", sshape, out_s.tobytes()),
        prefix + ".biases": ("BF16", sshape, out_b.tobytes()),
    }


def verify_module(model: str, index: dict, name: str, spec: dict, dst_bits: int,
                  tensors: dict[str, tuple[str, tuple, bytes]], limit_rows: int = 8) -> tuple[float, float]:
    prefix = "language_model." + name
    weight = read_tensor(model, index, prefix + ".weight")
    scales = read_tensor(model, index, prefix + ".scales")
    src_bits = int(spec["bits"])
    k = weight.shape[-1] * (4 if src_bits == 8 else 8)
    flat = weight.reshape(-1, weight.shape[-1])[:limit_rows]
    sflat = scales.reshape(-1, scales.shape[-1])[:limit_rows]
    ref = (decode_mxfp8 if src_bits == 8 else decode_mxfp4)(flat, sflat)
    q = np.frombuffer(tensors[prefix + ".weight"][2], dtype=np.uint32).reshape(-1, k * dst_bits // 32)[:limit_rows]
    s = np.frombuffer(tensors[prefix + ".scales"][2], dtype=np.uint16).reshape(-1, k // DST_GROUP)[:limit_rows]
    b = np.frombuffer(tensors[prefix + ".biases"][2], dtype=np.uint16).reshape(-1, k // DST_GROUP)[:limit_rows]
    if dst_bits == 8:
        codes = unpack_codes(q, 8, k)[:limit_rows].astype(np.float32)
    else:
        codes = unpack_codes(q, dst_bits, k)[:limit_rows].astype(np.float32)
    sf = s.view(np.uint16).astype(np.float32)
    bf = b.view(np.uint16).astype(np.float32)
    # bf16 bits -> f32
    sf = (sf.astype(np.uint32) << 16).view(np.float32)
    bf = (bf.astype(np.uint32) << 16).view(np.float32)
    got = codes.reshape(-1, k // DST_GROUP, DST_GROUP) * sf[:, :, None] + bf[:, :, None]
    got = got.reshape(-1, k)
    diff = got - ref
    return float(np.abs(diff).max()), float(np.sqrt((diff.astype(np.float64) ** 2).mean()))


def update_config(config: dict, converted: dict[str, int]) -> None:
    """Mark every converted module affine (and drop it from non_affine_modules)."""
    def spec(bits: int) -> dict:
        return {"bits": bits, "group_size": DST_GROUP, "mode": "affine", "quantize_input": True}

    for key in ("quantization", "quantization_config"):
        q = config.get(key)
        if not isinstance(q, dict):
            continue
        non = q.get("non_affine_modules")
        if isinstance(non, dict):
            for name, bits in converted.items():
                non.pop(name, None)
                q[name] = spec(bits)
    # The translated config duplicates the block in text_config; keep both in step.
    text = config.get("text_config")
    if isinstance(text, dict):
        for key in ("quantization", "quantization_config"):
            q = text.get(key)
            if not isinstance(q, dict):
                continue
            non = q.get("non_affine_modules")
            if isinstance(non, dict):
                for name, bits in converted.items():
                    non.pop(name, None)
                    q[name] = spec(bits)
    omlx = config.get("omlx_deepseek_v41")
    if isinstance(omlx, dict):
        qm = omlx.get("quantized_modules")
        if isinstance(qm, dict):
            for name, bits in converted.items():
                qm["language_model." + name] = spec(bits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--verify", action="store_true", help="compare a few rows of each conversion")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    model = os.path.abspath(args.model_dir)
    out = os.path.abspath(args.out_dir)
    if os.path.exists(out) and not args.overwrite:
        print(f"{out} exists (use --overwrite)", file=sys.stderr)
        return 1
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(model, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(model, "model.safetensors.index.json"), encoding="utf-8") as f:
        index = json.load(f)

    todo = modules(config)
    if not todo:
        print("no DSpark MXFP modules found", file=sys.stderr)
        return 1
    print(f"{len(todo)} DSpark modules:")
    for name, spec in sorted(todo.items()):
        print(f"  {name}: mxfp{spec['bits']}")

    tensors: dict[str, tuple[str, tuple, bytes]] = {}
    converted: dict[str, int] = {}
    for name, spec in sorted(todo.items(), key=lambda kv: (".experts." not in kv[0], kv[0])):
        dst_bits = 3 if ".ffn.experts." in name else 8
        print(f"converting {name} -> affine-{dst_bits}/g{DST_GROUP} ...", flush=True)
        converted[name] = dst_bits
        tensors.update(convert_module(model, index, name, spec, dst_bits))
        if args.verify:
            mx, rms = verify_module(model, index, name, spec, dst_bits, tensors)
            print(f"  verify: max |diff| {mx:.6f}, rms {rms:.6f}")

    # Hardlink the payload shards and assets, but never the two metadata files:
    # they are rewritten below, and a hardlink would edit the source directory.
    for entry in os.listdir(model):
        src = os.path.join(model, entry)
        dst = os.path.join(out, entry)
        if os.path.exists(dst):
            continue
        if entry in ("config.json", "model.safetensors.index.json"):
            shutil.copyfile(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            os.link(src, dst)
    shard = "model-dspark-affine.safetensors"
    data = b"".join(body for _, _, body in tensors.values())
    header: dict[str, dict] = {"__metadata__": {"format": "pt", "converted": "dspark-affine"}}
    off = 0
    for name, (dtype, shape, body) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(body)]}
        off += len(body)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    with open(os.path.join(out, shard), "wb") as f:
        f.write(struct.pack("<Q", len(raw_header)))
        f.write(raw_header)
        f.write(data)
    index["weight_map"].update({name: shard for name in tensors})
    update_config(config, converted)
    with open(os.path.join(out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"wrote {os.path.join(out, shard)}: {len(tensors)} tensors, {len(data) / GIB:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
