#!/usr/bin/env python3
"""Layer-level reference for the MLX affine checkpoint, on the recipe's PyTorch model.

The recipe's `engine/reference` implements DeepSeek-V4.1's block math (mHC, CSA2 attention,
router + experts, Engram) for the EXL3 pack. This adapter lets the same reference run against
the MLX affine checkpoint: it reads the checkpoint's `language_model.*` tensors, dequantizes
each affine projection (U32 words, BF16 scales/biases, group 64) to an fp32 `[out, in]`
matrix, slices the stacked experts, and splits the dense `wo_a` into its per-output-group
linears. Native tensors stay exactly as stored (BF16/F32 values).

Run it in the PyTorch image with the recipe checkout at /recipe, the checkpoint at /model and
an output directory at /out:

  python3 reference_affine_layer.py 671,6102,294,8760,344 --layers 1

Writes /out/ref_layer0.npy (and /out/ref_logits.npy with --head): [T, hc, hidden] fp32 from
the last requested layer, for comparison with the engine's traced per-layer streams.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import torch

RECIPE = "/recipe/engine"
if RECIPE not in sys.path:
    sys.path.insert(0, RECIPE)

from reference.attention import CsaState  # noqa: E402
from reference.config import Config  # noqa: E402
from reference.hc import hc_post  # noqa: E402
from reference.loader import CheckpointLoader, DTYPES, SIZES, SafetensorsDir  # noqa: E402
from reference.model import Block, Model  # noqa: E402
from reference.ops import DenseLinear, F32, Numerics, bf16  # noqa: E402

DTYPES["U32"] = torch.int32
SIZES["U32"] = 4

PREFIX = "language_model."


class Prefixed:
    """A safetensors source whose checkpoint names carry the `language_model.` prefix."""

    def __init__(self, src: SafetensorsDir):
        self.src = src

    def __contains__(self, name: str) -> bool:
        return (PREFIX + name) in self.src

    def names(self):
        return [n[len(PREFIX):] for n in self.src.names() if n.startswith(PREFIX)]

    def tensor(self, name: str):
        return self.src.tensor(PREFIX + name)

    def info(self, name: str):
        return self.src.info(PREFIX + name)

    def rows(self, name: str, ids):
        return self.src.rows(PREFIX + name, ids)

    def prefetch(self, *args, **kwargs):
        return None


def dequant(words, scales, biases, bits: int, group: int) -> torch.Tensor:
    """MLX affine `[N, packed U32]` + `[N, G]` bf16 scales/biases -> `[N, K]` fp32."""
    w = words.detach().cpu().numpy()
    raw = w.view(np.uint8).reshape(w.shape[0], -1)
    n, row_bytes = raw.shape
    k = row_bytes * 8 // bits
    bits_arr = np.unpackbits(raw, axis=1, bitorder="little")[:, : k * bits]
    shifts = (1 << np.arange(bits, dtype=np.uint32)).astype(np.uint32)
    codes = (bits_arr.reshape(n, k, bits) * shifts).sum(-1).astype(np.float32)
    s = scales.detach().cpu().float().numpy()
    b = biases.detach().cpu().float().numpy()
    groups = k // group
    out = codes.reshape(n, groups, group) * s[:, :, None] + b[:, :, None]
    return torch.from_numpy(out.reshape(n, k))


class AffineLoader(CheckpointLoader):
    def __init__(self, cfg: Config, src: Prefixed, quant: dict):
        super().__init__(cfg, src, Numerics.exact())
        self.quant = quant

    def is_exl3(self, prefix: str) -> bool:
        return False

    def _spec(self, prefix: str):
        spec = self.quant.get(prefix)
        if isinstance(spec, dict) and spec.get("bits") and spec.get("mode", "affine") == "affine":
            return spec
        return None

    def _affine(self, prefix: str):
        spec = self._spec(prefix)
        if spec is None:
            return None
        return dequant(
            self.src.tensor(prefix + ".weight"),
            self.src.tensor(prefix + ".scales"),
            self.src.tensor(prefix + ".biases"),
            int(spec["bits"]),
            int(spec.get("group_size", 64)),
        )

    def _expert(self, prefix: str):
        m = re.fullmatch(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)", prefix)
        if not m:
            return None
        layer, expert, proj = m.groups()
        base = f"layers.{layer}.ffn.experts.{proj}"
        spec = self._spec(base)
        if spec is None:
            return None
        return dequant(
            self.src.tensor(base + ".weight")[int(expert)],
            self.src.tensor(base + ".scales")[int(expert)],
            self.src.tensor(base + ".biases")[int(expert)],
            int(spec["bits"]),
            int(spec.get("group_size", 64)),
        )

    def linear(self, prefix: str, registry: list | None = None):
        # The reference takes wo_a as one linear per output group; the MLX checkpoint stores
        # the whole [o_groups * o_lora, per_group_k] matrix as one dense bf16 tensor.
        if ".wo_a.slice." in prefix:
            group = int(prefix.rsplit(".", 1)[1])
            base = prefix.rsplit(".slice.", 1)[0]
            w = self.src.tensor(base + ".weight").to(F32)
            g = self.cfg.o_groups
            return DenseLinear(w.reshape(g, w.shape[0] // g, w.shape[1])[group])
        d = self._expert(prefix)
        if d is None:
            d = self._affine(prefix)
        if d is not None:
            return DenseLinear(d)
        return super().linear(prefix, registry)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", help="comma-separated token ids (the engine's prompt ids)")
    ap.add_argument("--model", default="/model")
    ap.add_argument("--out", default="/out/ref_layer.npy")
    ap.add_argument("--layers", type=int, default=1, help="run this many blocks")
    ap.add_argument("--keep-layer", type=int, default=0, help="layer whose streams to save")
    ap.add_argument("--head", action="store_true", help="also run the final norm and head")
    ap.add_argument("--sublayers", action="store_true",
                    help="save layer 0's attention output, MoE output, their inputs and streams")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x]
    cfg = Config.from_file(args.model + "/config.json")
    raw = json.load(open(args.model + "/config.json"))
    quant = raw.get("quantization") or raw.get("quantization_config") or {}
    src = Prefixed(SafetensorsDir(args.model))
    loader = AffineLoader(cfg, src, quant)
    w = loader.model_weights(with_head=args.head, with_engram=False)
    if args.sublayers:
        lw = loader.layer(0, with_engram=False)
        blk = Block(cfg, 0, lw, Numerics.exact())
        h = bf16(loader.embed(torch.tensor(ids)))
        streams = h.unsqueeze(1).expand(-1, cfg.hc_mult, -1).contiguous()
        pos = torch.arange(len(ids))
        state = CsaState()
        post, comb, xa, pre_a = blk._pre(streams, lw.hc_attn, None, lw.attn_norm)
        a = blk.attn(xa, pos, state)
        s1 = hc_post(a, streams, post, comb)
        post2, comb2, xf, pre_f = blk._pre(s1, lw.hc_ffn, pre_a, lw.ffn_norm)
        y = blk.moe(xf)
        s2 = hc_post(y, s1, post2, comb2)
        saved = 0
        for name, t in (("attn", a), ("moe", y), ("xattn", xa), ("xmoe", xf), ("streams", s2)):
            arr = t.detach().to(torch.float32).numpy()
            np.save(f"/out/ref_{name}.npy", arr)
            print(f"{name} {arr.shape} rms {float(np.sqrt((arr.astype(np.float64) ** 2).mean())):.6f}")
            saved += 1
        return 0 if saved else 1
    model = Model(cfg, w)
    keep = (args.keep_layer,)
    out = model.forward(torch.tensor(ids), n_layers=args.layers, keep_layer_out=keep,
                        with_logits=args.head)
    streams = out.layer_out[args.keep_layer].to(torch.float32).numpy()
    np.save(args.out, streams)
    print(f"ids {ids} layers {args.layers}: streams {streams.shape} "
          f"rms {float(np.sqrt((streams.astype(np.float64) ** 2).mean())):.6f} "
          f"max {float(np.abs(streams).max()):.6f} -> {args.out}")
    if args.head and out.logits is not None:
        logits = out.logits.to(torch.float32).numpy()
        np.save(args.out.replace(".npy", "_logits.npy"), logits)
        print(f"logits {logits.shape} argmax {int(logits[0].argmax())} -> {args.out.replace('.npy', '_logits.npy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
