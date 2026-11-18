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
| `0028-dsv41-affine-budget.patch` | Log raw/packed affine bytes and the largest packed expert at boot |
| `0029-dsv41-affine-linears.patch` | Pack every affine linear into a resident name-addressed device set (`TF_DSV41_AFFINE_LOAD=1`) |
| `0030-dsv41-affine-linear-dev.patch` | Device-to-device `AffineN` linear with `act_rows.groupSums`; smoke checks the dev path |
| `0031-dsv41-attn-chain.patch` | Layer-0 attention projection chain: `attn_norm` → `wq_a` → `q_norm` → `wq_b`, `wkv` → `kv_norm` |
| `0032-dsv41-mxfp4-requant.patch` | Requantize MXFP4 rows to affine-4/group-64 for the native expert kernels |
| `0033-dsv41-affine-emitter.patch` | Affine runtime (modules, act rows, resident packed linears), the emitter call site, the dense/native plan+loader (`wo_a`, head, norms, router), and the attention/dense projection selectors |
| `0034-dsv41-expert-host-loop.patch` | Raw routed-expert loading, the host-loop MoE correctness path (shared expert + per-expert `AffineN`), and compressor/indexer/Engram emitter coverage |
| `0035-dsv41-engram-native.patch` | Native 4-bit Engram reader streamed from SSD (160 B rows; the FP8 264 B format remains readable) |
| `0036-dsv41-affine-review-fixes.patch` | Resolve `language_model.*` dense names, unify `L<i>` affine roles, make prefetch/graphs/boot affine-safe, and add the load budget and deinit fixes |
| `0037-dsv41-activation-quant.patch` | Apply the MLX checkpoint's `quantize_input` FP8 E4M3 group-32 round-trip to affine linears, experts, and `wo_a`; parse `gate_temp` |
| `0038-dsv41-token-map-tool.patch` | Install `tf-dsv41-token-map`, the release tokenizer's Engram token-map CLI, with the serving pair |
| `0039-dsv41-smoke-quant.patch` | Round the smoke's projection inputs through the `quantize_input` FP8 path before the linear and expert comparisons |
| `0040-dsv41-metadata-experts.patch` | Let the metadata parse (asset tools) admit the release's pruned 192-expert count without a storage format |
| `0041-dsv41-dense-mark-spelling.patch` | Resolve coverage marks to the pack's module-prefixed spelling, so the affine dense plan loads `language_model.*` tensors |
| `0042-dsv41-aot-needs-affine.patch` | Optionally read the checkpoint config in `aot-needs`, so the affine decode programs (not the EXL3 widths/prefill set) are enumerated |
| `0043-dsv41-quant-rank.patch` | Accept the attention output's `[rows, heads, head_dim]` rank in the `wo_a` FP8 round-trip call |
| `0044-dsv41-plain-head-chunks.patch` | Launch the vocabulary-sized dense head in aligned 32,768-column `_plain` chunks (CUDA `grid.y` limit) |
| `0045-dsv41-affine-diagnostics.patch` | Name a missing affine role and a failing Triton call (grid, args) instead of returning a bare error |
| `0046-dsv41-index-grid.patch` | Run the indexer's `_scores` with at least one program when a KV source has no compressed keys yet |
| `0047-dsv41-affine-trace-env.patch` | Register `TF_DSV41_AFFINE_TRACE` in the strict environment table |
| `0048-dsv41-affine-miss-trace.patch` | Log the missing affine call id and, under the trace, the plan's packed/skipped modules |
| `0049-dsv41-engram-wkv-shape.patch` | Carry `engram.wkv`'s hash-column/gate shape in the release block list so the affine plan packs it |
| `0050-dsv41-layer-stats.patch` | Log each layer's traced residual RMS/max/mean under `TF_DSV41_LAYER_STATS` for divergence bisecting |
| `0051-dsv41-smoke-2row.patch` | Smoke a two-row affine launch against the CPU dots (the server's prefill path) |
| `0052-dsv41-trace-dump.patch` | Dump the traced head input/logits (and per-layer states) under `TF_DSV41_TRACE_DIR` |
| `0053-dsv41-sublayer-trace.patch` | Trace the attention output and MoE output (dtype-aware dumps) for sublayer bisecting |
| `0054-dsv41-affine-mhc-forms.patch` | Build the persistent bf16 mHC `fn16` forms for the affine path (they were only built for EXL3) |
| `0055-dsv41-affine-dspark.patch` | Serve DSpark from the affine model: dense DSpark plan/heads, affine `main_proj`/attention/head, host expert MoE |
| `0056-dsv41-pack-prefix-has.patch` | Make `Pack.has` find prefixed (`language_model.`) tensors so the dense DSpark heads plan |
| `0057-dsv41-dspark-forward-glue.patch` | Route DSpark's shared glue steps (`copy_cols`, `moe_host`) to the forward glue |
| `0058-dsv41-affine-expert-tiles.patch` | Pack every affine expert into resident `AffineN` tiles at load; the host loop launches them directly |
| `0059-dsv41-affine-moe-batched.patch` | Batch the affine MoE: one activation quant/sums, device expert launches, and two host transfers per layer |
| `0060-dsv41-affine-moe-host-quant.patch` | Use the host's FP8 round-trip and the original combine order in the batched MoE (bit parity with the tested path) |
| `0061-dsv41-fused-woa.patch` | Fuse wo_a's eight block-diagonal BF16 groups into one glue kernel (removes 8 launches a layer) |
| `0062-dsv41-logits-trace.patch` | Number every decode step's head-logits trace so the correctness rule can score a whole request |
| `0063-dsv41-device-moe-post.patch` | Keep activation FP8 quantization, clamped SwiGLU, and the shared+routed combine on the device |
| `0064-dsv41-moe-host-controls.patch` | Temporary per-post-op host fallbacks used to bisect the one-prompt SwiGLU mismatch |
| `0065-swiglu-near-midpoint.patch` | Recompute SwiGLU's exp in double only when the fp32 product is within 32 ULPs of a bf16 midpoint (exact host parity) |
| `0066-dsv41-gathered-experts.patch` | Gather a layer's routed experts from the device pick in three launches; cache the MoE scratch buffers |
| `0067-dsv41-woa-block16.patch` | Compute 16 `wo_a` outputs per block, keeping the original strided per-lane summation order |
| `0068-dsv41-moe-graphs.patch` | Preallocate the MoE scratch and let CUDA graph capture include the device-only `moe_host` |
| `0069-dsv41-head-gemv.patch` | Replace the fp64 `_plain` head with a bf16 CUDA GEMV writing fp32 logits |`n| `0070-dsv41-splitk-gb10.patch` | Enable the existing split-K linear kernels on GB10 |
| `0071-dsv41-expert-split8.patch` | Deepen K slices in the one-row expert GEMV |

Implemented: `0001`-`0071`.

Rules:

- One logical change per patch; apply cleanly with `git apply --check`.
- Rebase onto the pinned `TF_REF` before adding a new patch; do not stack unrelated changes.
- Keep the upstream file layout; put new family code under `zig/src/families/deepseek_v41/`.
- Record the upstream commit and the patch set in `Agents/Implementation.md`.
