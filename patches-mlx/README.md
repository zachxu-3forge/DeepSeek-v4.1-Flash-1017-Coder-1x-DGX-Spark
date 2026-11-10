# patches-mlx

Engine patches applied in filename order to upstream TensorFold (`TF_REF` in
`docker/Dockerfile.mlx`). The current set:

| Patch | Purpose |
| --- | --- |
| `0001-admit-omlx-affine.patch` | Read `omlx_deepseek_v41.quantized_modules`, translate it to the engine's affine block, and admit `quant_method = "affine"` in the dsv41 config path |
| `0002-dsv41-affine-layout.patch` | Resolve translated affine metadata and validate packed U32/scales/biases shapes (host-only) |
| `0003-dsv41-affine-tensor-reader.patch` | Read U32/BF16 named images, preserve stacked experts, and keep the whole `wo_a` |
| `0004-dsv41-affine-plan.patch` | Enumerate affine linears and expert stacks from headers; default unlisted modules to dense |
| `0005-dsv41-affine-boot-guard.patch` | Build and validate the affine plan at boot; return `AffineForwardNotPorted` before the EXL3 path |
| `0006-dsv41-192-world1.patch` | Exercise a 192-expert affine plan and single-rank admission |
| `0007-dsv41-mixed-dspark.patch` | List DSpark MXFP8/MXFP4 modules as mixed and log them at boot |
| `0008-dsv41-affine-reference.patch` | CPU reference for the MLX bit-stream dequantization and row dot |
| `0009-dsv41-affine-cuda-audit.patch` | Report native CUDA width support; 2-bit routed experts need a per-expert fallback |
| `0010-dsv41-expert-fallback.patch` | Per-expert byte views into stacked affine projections for the `AffineN` fallback |
| `0011-dsv41-mixed-reader.patch` | Read MXFP8/MXFP4 DSpark U32 words and U8 E8M0 scales |
| `0012-dsv41-affine-dispatch.patch` | Build the forward dispatch table: affine8 linears, per-expert fallback, MXFP8/MXFP4 |
| `0013-dsv41-bindings.patch` | Freeze affine/mixed layouts for the dispatch entries; bind device buffers later |
| `0014-dsv41-device-bindings.patch` | Bind the dispatch layouts to loaded device tensors with byte-size checks |
| `0015-dsv41-one-layer-selection.patch` | Make the affine boot guard check only `TF_DSV41_LAYERS`, enabling one-layer fixtures |
| `0016-dsv41-affine-pack.patch` | Pack affine projections into `AffineN` device tiles; no kernel launch yet |
| `0017-dsv41-affine-launch.patch` | Launch `AffineNKernels.decode`/`prompt` with x, group sums, and output buffers |
| `0018-dsv41-affine-smoke.patch` | One-projection smoke test: load `wq_a`, pack, decode two rows, log `y[0]` |
| `0019-dsv41-affine-smoke-env.patch` | Register `TF_DSV41_AFFINE_SMOKE` in the strict environment and profile tables |
| `0020-dsv41-pack-spelling.patch` | Resolve canonical plan names to the pack's `language_model.` spelling |
| `0021-dsv41-indexer-weights-proj.patch` | Plan the checkpoint's quantized `attn.indexer.weights_proj` as an affine linear |
| `0022-dsv41-affine-group-sums.patch` | Fill the affine decode's per-row 64-value group sums before launch |
| `0023-dsv41-smoke-oracle.patch` | Log the CPU oracle for the smoke row and zero the output buffer |
| `0024-dsv41-smoke-raw.patch` | Log the loaded row's raw words/scales/biases and tensor kinds |
| `0025-dsv41-expert-smoke.patch` | Run expert 0 of `layers.0.ffn.experts.w1` through the per-expert fallback and log `y[0]` |
| `0026-dsv41-smoke-6bit.patch` | Also smoke the 6-bit `layers.0.ffn.shared_experts.w1` linear |
| `0027-dsv41-layer0-sweep.patch` | Sweep every layer-0 linear and all three 2-bit expert stacks in the smoke |
| `0028-dsv41-world1.patch` | Admit and run the single-rank serving path end to end |
| `0029-dsv41-engram-affine.patch` | Optional native 4-bit Engram reader; the FP8 converter is the current path |

Implemented: `0001`-`0027`. Planned: `0028`-`0029`.

Rules:

- One logical change per patch; apply cleanly with `git apply --check`.
- Rebase onto the pinned `TF_REF` before adding a new patch; do not stack unrelated changes.
- Keep the upstream file layout; put new family code under `zig/src/families/deepseek_v41/`.
- Record the upstream commit and the patch set in `Agents/Implementation.md`.
