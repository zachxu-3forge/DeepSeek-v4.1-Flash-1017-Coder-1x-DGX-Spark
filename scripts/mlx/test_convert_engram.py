#!/usr/bin/env python3
"""Unit checks for the Engram converter's E4M3/E8M0 math."""

from __future__ import annotations

import numpy as np

import convert_engram as c


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
