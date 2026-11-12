#!/usr/bin/env python3
"""Build the dsv41 CUDA `rope.bin`: two f32 tables [rows, 64], window then compressed.

The window (sliding) table is plain RoPE at `rope_theta`. The compressed table is YaRN
over `compress_rope_theta`, blended as `_compute_yarn_parameters` does, and its cosines and
sines carry the YaRN attention scale `0.1 ln(factor) + 1` (the same convention as the
MiaAI Qwen3.8 recipe). Verify the frequencies against `core/rope_yarn.zig` and the target's
MLX reference before serving 600K.

Header: 16 bytes, `ROPE` + rows + row floats + reserved. `boot_mem.uploadRope` only skips
those 16 bytes; the tables follow.
"""

from __future__ import annotations

import argparse
import math
import struct
from array import array

DIM = 64
HEADER = 16


def window_freqs(theta: float, dim: int = DIM) -> list[float]:
    return [1.0 / (theta ** (2.0 * i / dim)) for i in range(dim // 2)]


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def compressed_freqs(
    theta: float,
    factor: float,
    original: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    dim: int = DIM,
) -> list[float]:
    plain = window_freqs(theta, dim)
    if factor <= 1.0:
        return plain
    low = max(math.floor(dim * math.log(original / (beta_fast * 2.0 * math.pi)) / (2.0 * math.log(theta))), 0)
    high = min(math.ceil(dim * math.log(original / (beta_slow * 2.0 * math.pi)) / (2.0 * math.log(theta))), dim - 1)
    if low == high:
        high += 0.001
    out = []
    factor32 = _f32(factor)
    for i, f in enumerate(plain):
        # Mirror core/rope_yarn.zig's f32 order, not a final cast of f64 math.
        pos_freq = _f32(theta ** _f32(2.0 * i / dim))
        extra = _f32(1.0 / pos_freq)
        inter = _f32(1.0 / _f32(factor32 * pos_freq))
        keep = _f32(1.0 - _f32(min(max((i - low) / (high - low), 0.0), 1.0)))
        out.append(_f32(_f32(inter * _f32(1.0 - keep)) + _f32(extra * keep)))
    return out


def mscale(factor: float) -> float:
    return 1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0


def table(freqs: list[float], rows: int, scale: float = 1.0) -> array:
    if len(freqs) != DIM // 2:
        raise ValueError(f"expected {DIM // 2} frequencies, got {len(freqs)}")
    out = array("f")
    for pos in range(rows):
        for f in freqs:
            angle = pos * f
            out.append(math.cos(angle) * scale)
        for f in freqs:
            angle = pos * f
            out.append(math.sin(angle) * scale)
    return out


def build(
    out_path: str,
    rows: int,
    theta: float = 10000.0,
    compress_theta: float = 160000.0,
    factor: float = 16.0,
    original: int = 65536,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    apply_mscale: bool = True,
) -> None:
    window = table(window_freqs(theta), rows)
    scale = mscale(factor) if apply_mscale else 1.0
    compressed = table(compressed_freqs(compress_theta, factor, original, beta_fast, beta_slow), rows, scale)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sIII", b"ROPE", rows, DIM, 0))
        window.tofile(f)
        compressed.tofile(f)
    print(
        f"wrote {out_path}: {rows} rows, window theta {theta}, compressed theta {compress_theta}, "
        f"YaRN factor {factor} over {original}, mscale {scale:.7f}"
    )


def self_test() -> int:
    # The first 32 frequencies and the scale from core/rope_yarn.zig's Qwen3.6 vector.
    torch = [
        0x3F800000, 0x3F1AB32B, 0x3EBAF81B, 0x3E61F835, 0x3E088D77, 0x3DA50956, 0x3D47763F, 0x3CF11177,
        0x3C91AD39, 0x3C301052, 0x3BD4CA15, 0x3B80967D, 0x3B1B690D, 0x3ABBD3ED, 0x3A6301E2, 0x39F8A364,
        0x3986B53E, 0x3910058B, 0x38975C0D, 0x381B7E06, 0x379AC368, 0x3712F6ED, 0x36812DAB, 0x361C1FC4,
        0x35BCB0C1, 0x35640CC6, 0x3509CF4A, 0x34A68E4C, 0x34494C57, 0x33F3499D, 0x3393048E, 0x3331AF44,
    ]
    got = compressed_freqs(10000000.0, 4.0, 262144)
    for i, (g, want) in enumerate(zip(got, torch)):
        if struct.unpack("<I", struct.pack("<f", g))[0] != want:
            raise AssertionError(f"YaRN freq {i}: {g!r} != 0x{want:08x}")
    if abs(mscale(4.0) - 1.138629436111989) > 1e-12:
        raise AssertionError(mscale(4.0))

    # The target's spectra: plain window at theta, compressed at compress_theta/factor.
    window = window_freqs(10000.0)
    compressed = compressed_freqs(160000.0, 16.0, 65536)
    base = window_freqs(160000.0)
    if window[0] != 1.0 or compressed[0] != 1.0:
        raise AssertionError((window[0], compressed[0]))
    if abs(compressed[-1] - base[-1] / 16.0) > 1e-9:
        raise AssertionError((compressed[-1], base[-1] / 16.0))
    if abs(mscale(16.0) - 1.2772588722239782) > 1e-12:
        raise AssertionError(mscale(16.0))
    print("rope tables: dsv41 spectra and mscale 16x are consistent")
    print("rope tables: YaRN frequencies and scale match core/rope_yarn.zig")
    return 0


def check(path: str, apply_mscale: bool = True) -> int:
    with open(path, "rb") as f:
        head = f.read(HEADER)
        if len(head) != HEADER:
            raise SystemExit(f"{path}: short header")
        magic, rows, dim, reserved = struct.unpack("<4sIII", head)
        if magic != b"ROPE" or dim != DIM or reserved != 0 or rows == 0:
            raise SystemExit(f"{path}: bad header {magic!r} rows {rows} dim {dim}")
        table = array("f")
        table.fromfile(f, 2 * rows * DIM)
        if f.read(1):
            raise SystemExit(f"{path}: trailing bytes")
    if len(table) != 2 * rows * DIM:
        raise SystemExit(f"{path}: short table")
    window = window_freqs(10000.0)
    compressed = compressed_freqs(160000.0, 16.0, 65536)
    scale = mscale(16.0) if apply_mscale else 1.0
    picks = sorted({0, 1, rows // 2, rows - 1})
    for pos in picks:
        for i in (0, 1, DIM // 2 - 1):
            w = table[pos * DIM + i]
            c = table[rows * DIM + pos * DIM + i]
            want_w = math.cos(pos * window[i])
            want_c = math.cos(pos * compressed[i]) * scale
            if abs(w - want_w) > 1e-5 or abs(c - want_c) > 1e-5:
                raise SystemExit(f"{path}: row {pos} pair {i}: {w}/{c} != {want_w}/{want_c}")
    print(f"{path}: {rows} rows, window + YaRN compressed tables, mscale {scale:.7f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", help="write rope.bin here")
    ap.add_argument("--check", help="validate an existing rope.bin instead of writing")
    ap.add_argument("--rows", type=int, default=602048, help="positions (600K + a 2,048-row prefill segment)")
    ap.add_argument("--theta", type=float, default=10000.0)
    ap.add_argument("--compress-theta", type=float, default=160000.0)
    ap.add_argument("--factor", type=float, default=16.0)
    ap.add_argument("--original", type=int, default=65536)
    ap.add_argument("--beta-fast", type=float, default=32.0)
    ap.add_argument("--beta-slow", type=float, default=1.0)
    ap.add_argument("--no-mscale", action="store_true", help="leave the compressed table unscaled (the target reference applies YaRN's attention scale itself)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.check:
        if args.out:
            ap.error("OUT and --check are exclusive")
        return check(args.check, not args.no_mscale)
    if not args.out:
        ap.error("OUT is required unless --self-test or --check is used")
    if args.rows <= 0:
        ap.error("--rows must be positive")
    build(args.out, args.rows, args.theta, args.compress_theta, args.factor, args.original, args.beta_fast, args.beta_slow, not args.no_mscale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
