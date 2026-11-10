# Blockers: running the MLX checkpoint on one DGX Spark

Status: **not feasible with any current TensorFold release or the pinned fork without engine work.**
This file records the verified blockers, the evidence, and the decisions they force.

## Target

`mbrown-uno/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-MLX`

- MLX affine, mixed per module: routed experts 2-bit/group 64, attention and shared experts
  6/8-bit/group 64, Engram tables 4-bit/group 32, `wo_a` bf16 rounded to MXFP8 values,
  embeddings/LM head/routers/norms unquantized.
- 192 routed experts per layer (pruned from 384), 6 active + 1 shared; DSpark 128 experts/top-3.
- 21 shards, 211.4 GiB (227 GB decimal); shards 18 and 19 are the ~114 GiB Engram tables.
- Config carries `omlx_deepseek_v41.quantized_modules`; there is no `quantization_config`
  and no legacy `quantization` key.
- Model card: 97-99 GB resident, Engram streamed from SSD, Apple silicon/omlx only;
  "For CUDA or llama.cpp, use the original GGUF."

## Blockers

### B1. No MLX affine loader or kernels for the DeepSeek-V4.1 family

The pinned fork (`bertholomus/TensorFold`, branch `deepseek-v41-zig`, `fbbc971`) is EXL3-only:

- `native.zig`: `formats = &.{"exl3-mul1"}`.
- `weights.zig` and `rank_cache.zig`: every linear is an EXL3 trellis `tr/suh/svh`;
  dtypes are only bf16/f16/f32/i16/i32/i64/u8/f8e4m3. There is no packed-U32 affine layout.
- `config.zig`: requires `quantization_config.quant_method == "exl3"` and codebook `mul1`.

Upstream TensorFold 1.2.5 has MLX affine 2/3/4/5/6/8-bit CUDA readers, but only for the
Qwen dense family (and 4-bit/group-64 for Nemotron). Its DeepSeek-V4.1 family is different:

- `zig/src/families/deepseek_v41/config.zig`: `Format = enum { exl3, release_fp8 }`.
- `zig/src/families/deepseek_v41/serve_engine.zig`: `.formats = &.{"exl3"}`.
- Upstream README: the DeepSeek-V4.1 family is not served by any release binary;
  gate PR #827 is closed/unmerged.

No version has both halves: dsv41 + affine. The target's omlx-specific config key means even a
generic affine reader would need a config translator.

### B2. Expert tensors are stacked, not per-expert

The target index has one tensor per projection for all 192 experts, for example:

```
language_model.layers.0.ffn.experts.w1.weight
language_model.layers.0.ffn.experts.w1.scales
language_model.layers.0.ffn.experts.w1.biases
```

The header shows `[192, 2304, 320]` U32 with `[192, 2304, 80]` BF16 scales and biases:
192 experts, N=2304, K=5120 at 2-bit/group 64 (`packed_last = K * bits / 32`).

The pinned engine expects per-expert EXL3 groups (`layers.N.ffn.experts.E.w1.trellis`, `.suh`,
`.svh`) with pointer, width and scale tables. This needs a new loader/layout adapter and likely
new 2-bit affine expert-gather kernels; a name remap is not enough.

### B3. 192 experts are refused

- Pinned `config.zig` `checkShapes` hardcodes `experts == 384`.
- Upstream `validateRelease` compares `experts` to the 384-expert release and returns
  `error.UnsupportedArchitecture` for 192.

Routing tables, per-layer expert layouts, DSpark's 128-expert blocks, and memory admission all
assume the release shape.

### B4. TP=1 is refused by the pinned engine

`native.zig` accepts only world 2 or 4; `TF_TP_WORLD=1` returns `error.NotPortedYet`.
The README and `scripts/config.sh` are hardcoded `TP=2`, despite the `1x` folder name.

Upstream's planner accepts world 1 structurally and has world-3/world-4 tests, but there is no
released or qualified single-GPU DeepSeek-V4.1 path. A single-Spark run needs the rank/comm path
ported and tested, not just a config edit.

### B5. Engram storage format differs

The engine's packed Engram rows are FP8, not affine: `F8_E4M3 [rows,256]` plus
`F8_E8M0 [rows,8]`, 264 bytes a row, in `engram-l<layer>-r0of1.bin` shards with a
`DSV41EN1` header. The target stores U32 4-bit/group-32 words plus BF16 scales and
biases, 160 bytes a row, in shards 18/19 (~114 GiB). `scripts/mlx/convert_engram.py`
converts to the engine format (output ~203 GB, with E4M3 requantization loss); a native
4-bit reader would preserve the smaller tables but needs engine work. The `engram.json`
token map is still required either way.

The OMLX map lists `engram.wkv` but omits the `engram.embed` table. The table tensors are
present (`U32 [rows,32]`, `BF16 [rows,8]` scales/biases), so the converter and golden test
infer affine 4-bit/group-32 from the shapes. On 2026-11-10 the dry-run reported
188.83 GiB; a 64-row conversion verified with max 0.125 and RMS 0.0232/0.0256.

### B6. Mixed affine/MXFP8/MXFP4/unquantized modules

The target is not uniformly affine. Its omlx map has:

- 429 affine modules: 2-bit routed experts, 6-bit attention/shared, 8-bit attention/compressor/indexer.
- 22 MXFP8 modules: DSpark attention, DSpark shared experts, and `main_proj` in the three MTP blocks.
- 9 MXFP4 modules: DSpark routed experts (`w1`/`w2`/`w3` in each MTP block).
- Unquantized embeddings, LM head, routers, norms, and a bf16/MXFP8 `wo_a`.

The DSpark head is the full model's release-format head, not an affine quant. The dsv41 loader
must admit per-module affine, MXFP8, MXFP4, and dense formats. Upstream's `release_fp8` path
handles the release's FP8/MXFP4 blocks, and the Qwen affine reader handles affine, but neither
combines them in one dsv41 checkpoint.

### B7. Vision, tokenizer, chat template

Tokenizer and chat template are the same DeepSeek-V4.1 family, and the vision tower is copied
unchanged from the full MLX build. The engine has vision support, but the released DeepSeek
family does not serve, and the target's vision tensors are unquantized. Lower risk than B1-B5,
but not free.

### B8. Patch rebase is secondary

The 53 patches target `fbbc971`. Portable server/API patches (`0004`, `0027`-`0032`, `0036`,
`0046`-`0049`, `0057`, `0058`, `0064`-`0067`) can move to upstream 1.2.5. Engine/CUDA patches
depend on fork internals, TP2, and RDMA and need rework. Rebasing does not fix B1-B6.

### B9. CUDA/driver is not the blocker

Upstream 1.2.5 needs R580+/CUDA 13.0. The repo already builds on
`nvcr.io/nvidia/pytorch:26.07-py3` with Zig 0.17.0 and `sm_121`. Newer CUDA is a container
concern, not a host concern.

### B10. Runtime assets are not in the checkpoint

The engine expects `TF_DSV41_ASSETS`: RoPE tables, `engram.json`, and the related runtime
metadata. The target ships config, tokenizer, chat template, and safetensors only. The fork's
`kit/` is built for the EXL3 model; the dimensions and vocabulary match, so it may be reusable,
but this must be verified before a full-model load.

## Memory reality on one Spark

97-99 GB resident plus KV cache. A 128 GB Mac leaves about 7 GB headroom; DGX Spark OS/driver
overhead is comparable or worse. Short context may fit; 1,048,576 tokens will not. Engram must
stream from SSD, and the repo's existing Engram AIO/spill paths are for the base-model format.

## Consequence

- Serving the exact pruned MLX checkpoint with TensorFold is an engine port, not a recipe edit.
- The fastest way to run the model itself on one Spark is the source GGUF with llama.cpp CUDA.
- The existing two-Spark EXL3 recipe remains the known-working TensorFold deployment, but it
  serves the full 384-expert EXL3 checkpoint, not this pruned MLX one.
