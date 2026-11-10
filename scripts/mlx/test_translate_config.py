#!/usr/bin/env python3
"""Checks for `translate_config.py`'s non-affine output.

The OMLX map carries `group_size: null` for MXFP8/MXFP4. The engine's strict parser
treats a present null as a bad integer, so the translator must omit it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "omlx_deepseek_v41": {
                "quantized_modules": {
                    "language_model.layers.0.attn.wq_a": {
                        "bits": 8,
                        "group_size": 64,
                        "mode": "affine",
                    },
                    "language_model.mtp.0.attn.wq_a": {
                        "bits": 8,
                        "group_size": None,
                        "mode": "mxfp8",
                    },
                }
            }
        }
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
        out = os.path.join(tmp, "translated.json")
        subprocess.check_call(
            [sys.executable, os.path.join(here, "translate_config.py"), tmp, "--write-config", out],
            stdout=subprocess.DEVNULL,
        )
        translated = json.load(open(out, encoding="utf-8"))["quantization_config"]
        affine = translated["layers.0.attn.wq_a"]
        assert affine["bits"] == 8 and affine["group_size"] == 64, affine
        non = translated["non_affine_modules"]["mtp.0.attn.wq_a"]
        assert non == {"mode": "mxfp8", "bits": 8}, non
        assert translated["mtp.0.attn.wq_a"] is False
        print("translate config: non-affine nulls omitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
