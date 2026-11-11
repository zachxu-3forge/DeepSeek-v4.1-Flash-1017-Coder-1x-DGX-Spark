#!/usr/bin/env python3
"""MXFP8 E4M3 / MXFP4 E2M1 code points and the mixed decode path."""

from __future__ import annotations

import struct

import reference_layer as rl


def main() -> int:
    assert rl.e2m1(0) == 0.0
    assert rl.e2m1(1) == 0.5
    assert rl.e2m1(7) == 6.0
    assert rl.e2m1(9) == -0.5
    assert rl.e2m1(15) == -6.0

    assert rl.e4m3(0x00) == 0.0
    assert rl.e4m3(0x38) == 1.0
    assert rl.e4m3(0x40) == 2.0
    assert rl.e4m3(0x7E) == 448.0
    assert rl.e4m3(0xC8) == -4.0
    assert rl.e4m3(0x7F) != rl.e4m3(0x7F)  # NaN

    # One MXFP4 row: nibble 9 is -0.5, not 9; E8M0 127 is 1.0.
    words = struct.pack("<I", 0x00000009) + b"\0" * 28
    scales = bytes([127] * 2)
    out = rl.decode(words, scales, None, 4, 64, 32)
    assert out[0] == -0.5, out[0]
    assert out[1] == 0.0, out[1]
    print("MXFP4 E2M1 / MXFP8 E4M3 decode: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
