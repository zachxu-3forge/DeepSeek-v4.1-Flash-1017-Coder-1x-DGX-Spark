# Implementation Log

Living record of what was actually built, tried, measured, and decided while implementing
`Agents.md`. `Agents.md` is the plan; `Agents/Blockers.md` is the evidence for why the plan is
needed. This file is the history.

## How to update this file

- Add a dated entry to **Work log** for every implementation session.
- Record the exact command, the image/TensorFold revision, the result, and the artifact path.
- Put lasting choices in **Decisions**, with the alternative that was rejected and why.
- Update **Status** and **Phase tracker** after each session.
- Do not rewrite old entries; correct them with a new entry.
- Keep secrets, tokens, and full model paths out of this file.

## Status snapshot

| Field | Value |
|---|---|
| Current phase | Phase 1 - affine loader and layout adapter |
| Overall status | Stage 1p guard reaches AffineForwardNotPorted; image rebuild with 0021 pending |
| Base engine | Upstream `main` at `26e1b074b1512d645698eba269ccfb8ebc974c10` for Phase 0; fork fallback |
| CUDA base image | `nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c` |
| TensorFold ref | `26e1b074b1512d645698eba269ccfb8ebc974c10` |
| Zig version | 0.17.0 |
| Model revision | `mbrown-uno/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-MLX` at `c2afc6bb446f612ec0791e03834cdf4f8cb05583` (recorded 2026-11-09) |
| Container image | `tf-dsv41-mlx:dev`, ID `sha256:54c3f67c043aa2a0a5f4fc33ffb09307d841fdd4e6dc23f7602c0781a42164d6` |
| Last updated | 2026-11-10 |

## Phase tracker

| Phase | Status | Gate | Evidence |
|---|---|---|---|
| 0. Harness and admission spike | Done | Translated config + 192-expert, world-1 affine fixture loads | Image built; affine plan admits 192-expert/world-1; GPU guard pending |
| 1. Affine loader and layout adapter | In progress | One full layer matches dequantized reference | Stage 1p image built; GPU comparison pending |
| 2. Full model, short context | Not started | Text prompt served on one Spark; 32K stable | - |
| 3. Engram | Not started | Sampled lookups match reference; SSD streaming works | - |
| 4. Server, tools, vision | Not started | Chat completions, streaming, tools, one image request | - |
| 5. Qualification and docs | Not started | Metrics, regression suite, reproducible run record | - |

Status values: `Not started`, `In progress`, `Blocked`, `Done`, `Dropped`.

## Decisions

| ID | Date | Decision | Alternatives considered | Why |
|---|---|---|---|---|
| D0 | 2026-11-09 | Build and run entirely in Docker with `docker run --rm`; no host CUDA/TensorFold/Zig install | Host install | Host stays clean; reproducible image; model and Engram mount from host |
| D1 | 2026-11-09 | Attempt the exact MLX checkpoint; do not convert to EXL3 unless the port is abandoned | Convert MLX -> EXL3/FP8 | Exact checkpoint is the objective; conversion adds quality loss and still needs 192-expert/world-1 work |
| D2 | 2026-11-09 | Target 32K-64K context for the first working milestone; 1M later | Target 1M immediately | 97-99 GB resident leaves too little headroom on 128 GB |
| D3 | 2026-11-09 | Convert Engram to the engine format first; add an affine reader only if needed | Affine Engram reader first | Lower engine risk; dimensions and vocabulary already match |
| D4 | 2026-11-09 | Choose the base engine after the Phase 0 build/admission spike | Commit to upstream `main` or the pinned fork now | Upstream has affine readers but its dsv41 family is unreleased; the fork is proven but EXL3-only |
| D5 | 2026-11-09 | Use upstream `main` at `26e1b074b1512d645698eba269ccfb8ebc974c10` as the Phase 0 base | Pinned fork `fbbc971` | Upstream has MLX-affine CUDA readers and a dsv41 lane backend; the fork has neither affine nor world 1 |
| D6 | 2026-11-09 | Build `tensorfold-dsv41` and `tf-dsv41` with `zig build dsv41-serve`, passing both `-Dnvcc` and `-Ddsv41-nvcc` at `-Dsm=121` | Generic `install native` build | The dsv41 server is a separate install step; the generic server does not install the dsv41 pair |
| D7 | 2026-11-09 | Emit upstream-affine-compatible metadata: direct canonical per-module keys inside `quantization`, global fallback 4-bit/group 64 | A separate `modules` map | Matches `zig/src/families/qwen27/affine.zig` so the reader can be reused |
| D8 | 2026-11-09 | Default the CUDA base to `nvcr.io/nvidia/pytorch:26.07-py3` by tag until the digest is captured on the Spark | An unverified newer tag | 26.07 is the repo's known-good `sm_121` base; the digest is recorded in the first successful build |
| D9 | 2026-11-09 | Build and run with `--platform linux/arm64` by default | Let Docker pick the host platform | Prevents an accidental x86 image on a non-Spark build host; the Spark is arm64 |
| D10 | 2026-11-09 | Commit stages separately: stage 0 harness, stage 1a config admission, then stage 1b-1d layout/reader/plan | One large commit | Keeps each patch reviewable and reversible; multiple commits per stage are allowed |
| D11 | 2026-11-10 | Write the Stage 1 verification harness before 1m/1n/1o | Verify ad hoc later | Captures golden rows, Engram checks, and dispatch counts before the launch code changes |

## Work log

Newest entries first. Each entry: date, phase, what was done, command, result, artifacts,
next step.

### 2026-11-10 - Stage 1p Spark boot fixes

- **Phase:** 1.
- **Done:** first Spark boot attempts exposed four gaps before the affine guard could run:
  1. The translated overlay lacked `tokenizer.json`; the engine stopped with
     `tensorfold: cannot read /model/tokenizer.json`.
  2. The engine is gated: it needs `TF_EXPERIMENTAL=deepseek` and defaults to world 2, so
     `TF_TP_WORLD=1` was added to the harness.
  3. `translate_config.py` copied `"group_size": null` from the OMLX map for the 31
     MXFP8/MXFP4 modules. The engine's strict spec parser reads a present null as a bad
     integer and returned `BadAffine`; the translator now omits null fields.
  4. Patch `0018` added `TF_DSV41_AFFINE_SMOKE` without registering it in the strict
     environment table, so boot panicked with `unregistered strict environment parser`.
     `0019-dsv41-affine-smoke-env.patch` registers it in `platform/environment.zig` and
     `core/profile.zig`.
- **Also fixed:** the affine plan names modules canonically (`layers.0...`) while the pack
  spells them `language_model.layers.0...`; the plan's shared-expert probe and the loader's
  `pack.need` would miss. `0020-dsv41-pack-spelling.patch` adds `Pack.spellModule` and makes
  the plan carry the pack's spelling.
- **After 0020:** the guard reached `AffineForwardNotPorted` and logged 419 modules,
  30 mixed, `dispatch: 299 affine8 linears, 120 expert fallbacks, 21 mxfp8, 9 mxfp4`.
  The 10-linear gap was 8 quantized `attn.indexer.weights_proj` modules the plan did not
  enumerate, plus 2 affine `engram.wkv` tables it intentionally leaves to the Engram path.
  The 1-MXFP8 gap is DSpark's `main_proj`, which is handled by the DSpark plan, not the
  affine dispatch. `0021-dsv41-indexer-weights-proj.patch` plans `weights_proj`;
  `plan_dispatch.py` now reports `affine_engram` and `dspark_main_proj` separately, so the
  affine-plan expectation is **307/120/21/9**.
- **Verification:** `0001`-`0021` apply cleanly; ast-check passes. The overlay translator now
  emits no null `group_size`; a host plan simulation reports 496 enumerated modules, 30
  mixed, 0 errors.
- **Next:** rebuild `tf-dsv41-mlx:dev` with 0021, rerun the guard, then the smoke.

### 2026-11-10 - Spark image verified; harness dispatch/overlay fixes

- **Phase:** 0/1.
- **Done:** built `tf-dsv41-mlx:dev` from base digest
  `sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`; recorded image ID
  `sha256:54c3f67c043aa2a0a5f4fc33ffb09307d841fdd4e6dc23f7602c0781a42164d6` and labels
  `tf.ref=26e1b07...`, `tf.sm=121`. `tensorfold-dsv41 --version` prints `tensorfold-native 1.2.5`.
- **Harness fix:** `test_stage1.sh` no longer forces `TF_DSV41_LAYERS=0-0` for the guard run.
  The engine filters the dispatch count by requested layers, so the old run reported layer 0
  only (`2 affine8 linears, 3 expert fallbacks`) and could not match `plan_dispatch.json`.
  The guard returns before weight upload, so planning all layers is cheap and gives the full
  309/120/22/9 counts. The smoke run still uses `0-0`.
- **Overlay fix:** use hardlinks, not symlinks. A symlink to the host checkpoint dangles inside
  the container, where only the overlay is mounted at `/model`.
- **Verification:** host `plan_dispatch.py` confirms 460 quantized modules and counts
  309 affine8 linears, 120 affine 2-bit expert fallbacks, 22 MXFP8, 9 MXFP4.
- **Next:** host checks, translated overlay, container guard, one-layer fixture, affine smoke.

### 2026-11-10 - Stage 1p: affine smoke test

- **Phase:** 1.
- **Done:** added `patches-mlx/0018-dsv41-affine-smoke.patch` with `affine_smoke.zig`:
  under `TF_DSV41_AFFINE_SMOKE=1`, loads `layers.0.attn.wq_a`, uploads it, loads the
  `AffineN` kernels, packs the projection, and decodes two deterministic rows. Logs
  `affine smoke layers.0.attn.wq_a: y[0] ones=... alt=...`.
- **Approach used:** one projection only; the normal affine boot still returns
  `AffineForwardNotPorted`.
- **Verification:** `0018` applies on top of `0001`-`0017`; ast-check passes. The smoke
  needs the Spark build/GPU.
- **Artifacts:** `patches-mlx/0018-dsv41-affine-smoke.patch`.
- **Next:** run `CONTAINER=1 SMOKE=1 test_stage1.sh` and compare the two logged values
  with `reference_layer.json`.

### 2026-11-10 - Stage 1o: AffineN launch

- **Phase:** 1.
- **Done:** added `patches-mlx/0017-dsv41-affine-launch.patch` with `affine_launch.zig`:
  `linear`/`prompt` allocate x, group sums, and output, call
  `AffineNKernels.decode`/`prompt`, synchronize, and return the output buffer; `expert`
  packs one stacked expert and decodes it. `Output.download` reads the result back.
- **Approach used:** one projection at a time; no dsv41 emitter/forward wiring yet.
- **Verification:** `0017` applies on top of `0001`-`0016`; a stub-CUDA test exercises
  `linear` allocation and call.
- **Artifacts:** `patches-mlx/0017-dsv41-affine-launch.patch`.
- **Next:** wire a one-layer test that packs `attn.wq_a`, runs `linear`, and compares the
  output against the golden row.

### 2026-11-10 - Stage 1o-prep: AffineN packing

- **Phase:** 1.
- **Done:** added `patches-mlx/0016-dsv41-affine-pack.patch` with `affine_pack.zig`:
  `packedBytes`, `packLinear`, and `packExpert` allocate `AffineN` word/scale/bias
  buffers and call `AffineNKernels.packWords/packTable`; `Packed` owns and frees them.
- **Approach used:** the qwen3_5/spark25 `AffineN` pattern; no kernel launch yet.
- **Verification:** `0016` applies on top of `0001`-`0015`; `affine_pack` passes with a
  stub CUDA module; target tile sizes are non-zero.
- **Artifacts:** `patches-mlx/0016-dsv41-affine-pack.patch`.
- **Next:** stage 1o launch: call `AffineNKernels.decode/prompt` on one layer.

### 2026-11-10 - Engram converter

- **Phase:** 3 preparation.
- **Done:** added `scripts/mlx/convert_engram.py` and `test_convert_engram.py`. The
  converter reads the target's `U32 [rows,32]` 4-bit words plus BF16 scales/biases,
  dequantizes each 32-value group, requantizes to E4M3 with an E8M0 scale, and writes
  `engram-l<layer>-r0of1.bin` with the engine's 4096-byte `DSV41EN1` header.
- **Approach used:** FP8/E8M0 row shards, matching `engram_rows.zig` (256 E4M3 values +
  8 E8M0 scales = 264 bytes). `--limit-rows` and `--verify` allow a small run first.
- **Verification:** synthetic 4-row fixture round-trips with max/RMS 0; a random-value
  unit check reports ~0.079 RMS at scale ~3; output header/size checked.
- **Artifacts:** `scripts/mlx/convert_engram.py`, `scripts/mlx/test_convert_engram.py`.
- **Tradeoff:** full output is ~203 GB and E4M3 requantization is slightly lossy; a native
  4-bit Engram reader would keep ~114 GB but needs engine work. The `engram.json` token
  map is still required.
- **Next:** run `--limit-rows 64 --verify`, then the one-layer launch.

### 2026-11-10 - Stage 1prep: one-layer fixture and memory estimate

- **Phase:** 1 (preparation for 1o).
- **Done:** added `make_one_layer.py` (copies one layer + embed/norm/head into a translated
  fixture), `memory_estimate.py` (weights + 584 B/layer/token KV + reserve vs 128 GiB), and
  `patches-mlx/0015-dsv41-one-layer-selection.patch` so the affine guard checks only the
  layers selected by `TF_DSV41_LAYERS`.
- **Approach used:** one-layer fixture reduces a full-model load to a few GiB; the memory
  estimator uses the engine's KV record size and exposes the inputs for live calibration.
- **Verification:** `0015` applies on top of `0001`-`0014`; `make_one_layer.py --list` sees
  48 tensors for layer 0; the estimate puts 32K at ~112.7 GiB and 1M at ~134.8 GiB.
- **Artifacts:** `scripts/mlx/make_one_layer.py`, `scripts/mlx/memory_estimate.py`,
  `patches-mlx/0015-dsv41-one-layer-selection.patch`.
- **Next:** stage 1o; run the one-layer fixture on the Spark.

### 2026-11-10 - Stage 1n: device bindings

- **Phase:** 1.
- **Done:** added `patches-mlx/0014-dsv41-device-bindings.patch` with `affine_bind.zig`:
  looks up each dispatch entry's weight/scales/biases in a `get(name) ?{ptr,len}` map,
  checks byte sizes against the affine/mixed layout, and freezes device addresses,
  lengths, expert count, bits, and group.
- **Approach used:** generic map binding; no allocation or launch, so it is host-testable
  with fake tensors.
- **Verification:** `0014` applies on top of `0001`-`0013`; `affine_bind` passes in the
  26-test host suite.
- **Artifacts:** `patches-mlx/0014-dsv41-device-bindings.patch`.
- **Next:** stage 1o, launch the affine8 linears and per-expert fallback on one layer.

### 2026-11-10 - Stage 1m: binding layouts

- **Phase:** 1.
- **Done:** added `patches-mlx/0013-dsv41-bindings.patch` with `affine_exec.zig`: each
  dispatch entry freezes its affine or mixed layout (`n`, `k`, `bits`, `group`) for the
  eventual device binding. `affine_plan.Mixed` and `affine_dispatch.Entry` now carry `n`/`k`.
- **Approach used:** host-side binding descriptors; no device buffer allocation or launch.
- **Verification:** `0013` applies on top of `0001`-`0012`; `affine_exec` passes in the
  25-test host suite.
- **Artifacts:** `patches-mlx/0013-dsv41-bindings.patch`.
- **Next:** stage 1n, allocate/bind device buffers and launch the one-layer path.

### 2026-11-10 - Stage 1 test harness

- **Phase:** 1 (preparation for 1m/1n/1o).
- **Done:** added `plan_dispatch.py` (expected 309 affine8 linears, 120 expert fallbacks,
  22 MXFP8, 9 MXFP4), `reference_layer.py` (golden CPU dots for 8/6/2-bit and mixed modules),
  `test_engram.py` (golden dots and row counts), `test_stage1.sh` (orchestrator and summary),
  and `stage1_tests.md`. The container wrapper now passes `TF_DSV41_*`, `TF_TP_*`, `TF_DS_*`,
  and `TENSORFOLD_*` variables through.
- **Approach used:** independent Python bit-stream decoders, separate from the Zig reference.
- **Verification:** Python/shell syntax passes; `plan_dispatch.py` reproduces the published
  460-module split exactly. Full host run requires the model shards.
- **Artifacts:** `scripts/mlx/plan_dispatch.py`, `reference_layer.py`, `test_engram.py`,
  `test_stage1.sh`, `stage1_tests.md`.
- **Next:** stage 1m; run `test_stage1.sh` before and after.

### 2026-11-09 - Stage 1l: forward dispatch table

- **Phase:** 1.
- **Done:** added `patches-mlx/0012-dsv41-affine-dispatch.patch` with `affine_dispatch.zig`:
  each affine/mixed module maps to `.affine8_linear`, `.affine_n_expert`, `.mxfp8`, or
  `.mxfp4`; boot logs the counts.
- **Approach used:** host-side dispatch plan; no buffer binding or kernel launch yet.
- **Verification:** `0012` applies on top of `0001`-`0011`; `affine_dispatch` passes in the
  18-test host suite.
- **Artifacts:** `patches-mlx/0012-dsv41-affine-dispatch.patch`.
- **Next:** stage 1m, bind the dispatch entries to buffers and launch the one-layer path.

### 2026-11-09 - Stage 1k: MXFP8/MXFP4 reader

- **Phase:** 1.
- **Done:** added `patches-mlx/0011-dsv41-mixed-reader.patch`:
  `mixed.zig` validates MXFP8/MXFP4 layouts (U32 words, U8 E8M0 scales, group 32) and
  provides the dequantization/dot reference; `mixed_load.zig` reads the packed U32/U8
  tensors from the pack. `Named.Kind` gained `u8`.
- **Approach used:** discovered the MXFP8/MXFP4 modules are MLX bit-stream U32 words with
  U8 E8M0 scales and no bias, so the existing bit-stream code path is reused.
- **Verification:** `0011` applies on top of `0001`-`0010`; the 35-test host suite passes.
- **Artifacts:** `patches-mlx/0011-dsv41-mixed-reader.patch`.
- **Next:** stage 1l, wire the `AffineN` fallback loop and affine linears into the forward.

### 2026-11-09 - Stage 1j: per-expert fallback views

- **Phase:** 1.
- **Done:** added `patches-mlx/0010-dsv41-expert-fallback.patch` with `affine_cuda.ExpertView`
  and `expertView`: byte offsets into a stacked `[E, N, packed]` weight and its scales/biases,
  with bounds checks, for a per-expert `AffineN` call.
- **Approach used:** host-side view arithmetic; the actual qlinear launch loop is not wired.
- **Verification:** `0010` applies on top of `0001`-`0009`; `affine_cuda` passes in the
  19-test host suite.
- **Artifacts:** `patches-mlx/0010-dsv41-expert-fallback.patch`.
- **Next:** stage 1k, the MXFP8/MXFP4 DSpark reader; then the qlinear fallback loop.

### 2026-11-09 - Stage 1i: CUDA width audit

- **Phase:** 1.
- **Done:** added `patches-mlx/0009-dsv41-affine-cuda-audit.patch` with `affine_cuda.zig`:
  linears at 2/3/5/6/8-bit group 64 are served by `affine8.cu`; native expert kernels
  serve only affine-3/4, so the target's 2-bit routed experts need a per-expert `AffineN`
  fallback. Boot now logs native/fallback/unsupported counts.
- **Approach used:** explicit capability audit before any forward wiring.
- **Verification:** `0009` applies on top of `0001`-`0008`; `affine_cuda` passes in the
  18-test host suite.
- **Artifacts:** `patches-mlx/0009-dsv41-affine-cuda-audit.patch`.
- **Next:** stage 1j, per-expert `AffineN` fallback for 2-bit experts.

### 2026-11-09 - Stage 1h: affine reference math

- **Phase:** 1.
- **Done:** added `patches-mlx/0008-dsv41-affine-reference.patch` with `affine_ref.zig`:
  MLX's little-endian bit-stream code extraction, BF16 scale/bias dequantization, and a
  row dot reference for 2/3/4/5/6/8-bit.
- **Approach used:** CPU oracle only; no device code. The 6-bit boundary crossing is covered.
- **Verification:** `0008` applies on top of `0001`-`0007`; `affine_ref` passes 3 tests;
  `affine_plan` suite passes 17 tests.
- **Artifacts:** `patches-mlx/0008-dsv41-affine-reference.patch`.
- **Next:** stage 1i, map the images into the forward and add the CUDA gather/linear path.

### 2026-11-09 - Stage 1g: mixed DSpark planning

- **Phase:** 1.
- **Done:** added `patches-mlx/0007-dsv41-mixed-dspark.patch`: `affine_plan` now records
  MXFP8/MXFP4 modules in a `mixed` list instead of only skipping them; boot logs the mixed
  count. Added a test with `mtp.0.attn.wq_a` (MXFP8) and `mtp.0.ffn.experts.w1` (MXFP4).
- **Approach used:** plan and report only; no MXFP8/MXFP4 reader or kernel yet.
- **Verification:** `0007` applies on top of `0001`-`0006`; `affine_plan` suite passes 17
  tests.
- **Artifacts:** `patches-mlx/0007-dsv41-mixed-dspark.patch`.
- **Next:** stage 1h, map affine images into the forward; the MXFP8/MXFP4 reader follows.

### 2026-11-09 - Stage 1f: 192 experts and world 1

- **Phase:** 1.
- **Done:** added `patches-mlx/0006-dsv41-192-world1.patch`: an `affine_plan` test with
  192 routed experts, and a `block_admission` test proving a single rank passes both
  serving and native-prefill admission.
- **Approach used:** tests only; no new runtime branch. The affine plan already uses
  `cfg.experts`, so the 192-expert path is exercised rather than hardcoded.
- **Verification:** `0006` applies on top of `0001`-`0005`; `block_admission` 9 tests pass;
  the `affine_plan` suite passes 16 tests.
- **Artifacts:** `patches-mlx/0006-dsv41-192-world1.patch`.
- **Next:** stage 1g, mixed MXFP8/MXFP4 DSpark handling; then forward mapping.

### 2026-11-09 - Stage 1e: affine boot guard

- **Phase:** 1.
- **Done:** added `patches-mlx/0005-dsv41-affine-boot-guard.patch`: at boot, affine
  checkpoints parse the translated map, build the affine plan, validate every module's
  headers with `affine_load.checkModule`, log layer/module/skipped counts, and return
  `AffineForwardNotPorted` before any EXL3 plan or weight upload.
- **Approach used:** explicit guard rather than a half-wired forward; the EXL3 path is
  untouched.
- **Verification:** `0005` applies on top of `0001`-`0004`; `ast-check` passes; the 15-test
  host suite still passes.
- **Artifacts:** `patches-mlx/0005-dsv41-affine-boot-guard.patch`.
- **Next:** stage 1f, data-driven 192-expert shapes and the single-rank serving path; then
  map the affine images into the forward.

### 2026-11-09 - Stage 1d: affine plan enumeration

- **Phase:** 1.
- **Done:** added `patches-mlx/0004-dsv41-affine-plan.patch` with `affine_plan.zig`:
  enumerates each layer's affine linears and routed expert stacks from pack headers,
  resolves per-module specs, records packed layouts, and lists dense/non-affine/Engram
  modules as skipped. `affine.zig` gained `dense_by_default`.
- **Approach used:** header-only plan; no payload reads, no device upload, no forward
  binding yet.
- **Verification:** `0004` applies on top of `0001`-`0003`; `affine_plan` test passes with
  the 15-test host suite (`config`, `affine`, `pack`, `names`, DSQ admission).
- **Artifacts:** `patches-mlx/0004-dsv41-affine-plan.patch`.
- **Next:** stage 1e, bind the plan and reader into `cuda_model`/`cuda_weights`.

### 2026-11-09 - Stage 1c: affine tensor reader

- **Phase:** 1.
- **Done:** added `patches-mlx/0003-dsv41-affine-tensor-reader.patch` with
  `affine_load.zig`: reads `.weight`/`.scales`/`.biases` from the pack into named host
  images, keeps U32/BF16 bytes unchanged, preserves the stacked expert axis, and validates
  shapes through `affine.Layout`. `named.Kind` gained `u32` and `Named.deinit`.
- **Approach used:** byte-preserving host reader; no dequantization, no expert split, no
  device upload or kernel selection yet.
- **Verification:** `0003` applies on top of `0001`+`0002`; `ast-check` passes for the new
  files; `config.zig` and `affine.zig` unit tests still pass. The full engine build is
  deferred.
- **Artifacts:** `patches-mlx/0003-dsv41-affine-tensor-reader.patch`.
- **Next:** stage 1d, wire the affine plan into `cuda_model`/`cuda_weights`.

### 2026-11-09 - Stage 1a fix: expert-count admission

- **Phase:** 1.
- **Done:** replaced the affine `continue` in `validateRelease` with a runtime `keep`
  condition. The old form was comptime control flow inside an `inline for` and would not
  compile; the new form skips only the `experts` comparison for affine packs.
- **Verification:** `zig test config.zig` passes all 8 tests, including the existing DSQ
  admission tests.
- **Artifacts:** updated `patches-mlx/0001-admit-omlx-affine.patch`.
- **Next:** continue stage 1c.

### 2026-11-09 - Stage 1b: affine layout parser

- **Phase:** 1.
- **Done:** added `patches-mlx/0002-dsv41-affine-layout.patch` with
  `zig/src/families/deepseek_v41/affine.zig`: translated-map resolution, per-module specs,
  MXFP8/MXFP4 separation, packed-word/group shape checks, and the `[N, packed]` /
  `[E, N, packed]` rules. `plan.zig` imports the module in its test block.
- **Approach used:** host-only parser and shape validation; no tensor reader or kernel wiring
  yet.
- **Verification:** Zig 0.17.0 `zig test` passes both unit tests; the patch applies cleanly on
  top of `0001` in a fresh worktree at `26e1b074`.
- **Artifacts:** `patches-mlx/0002-dsv41-affine-layout.patch`.
- **Next:** stage 1c, a raw affine tensor reader that emits U32/BF16 named images and the
  stacked-expert adapter.

### 2026-11-09 - Stage 1a: affine config admission

- **Phase:** 1.
- **Done:** added `patches-mlx/0001-admit-omlx-affine.patch`: `Format.affine`, `requireAffine`
  validation, an affine exception for the routed-expert count in `validateRelease`, and
  `"affine"` in the server family's format list.
- **Approach used:** accept the translated `quantization_config.quant_method = "affine"`,
  validate the global bits/group and the `non_affine_modules` modes, and keep every other
  release shape check.
- **Verification:** `git apply --check` against the pinned upstream tree passes. Engine compile
  and kernel verification are deferred to the end, per instruction.
- **Artifacts:** `patches-mlx/0001-admit-omlx-affine.patch`.
- **Next:** stage 1b, the affine layout parser and loader scaffold.

### 2026-11-09 - Phase 0 harness

- **Phase:** 0.
- **Done:** created `docker/Dockerfile.mlx` (upstream `dsv41-serve`, Zig 0.17.0, `sm_121`),
  `.dockerignore`, `patches-mlx/README.md`, `scripts/mlx/inspect_checkpoint.py`,
  `scripts/mlx/translate_config.py`, and `scripts/mlx/run_container.sh`.
- **Approach used:** upstream TensorFold `main` at `26e1b074...`; `zig build dsv41-serve` with
  both `-Dnvcc` and `-Ddsv41-nvcc`; translated config emits direct canonical per-module keys
  compatible with `zig/src/families/qwen27/affine.zig`.
- **Real config check:** translated the published config: 460 modules, 429 affine
  (2/6/8-bit), 22 MXFP8, and 9 MXFP4. The translator emits `false` for non-affine modules and
  records them under `non_affine_modules`; the DSpark head is not affine.
- **Published-index check:** the inspector sees 21 shards, 2,286 tensors, 211.4 GiB, 378
  stacked expert tensors, and 16 Engram tensors; `translate_config.py --check` reports every
  quantized piece present.
- **Commands:** `python -m py_compile scripts/mlx/*.py`; `bash -n scripts/mlx/run_container.sh`;
  synthetic-fixture run of both Python scripts.
- **Result:** Python and shell syntax pass; the inspector and translator run on a synthetic
  192-expert, 2/4/8-bit fixture. Docker is not installed on the local Windows host, so the
  image build and `dsv41-serve` compile are still unverified.
- **Artifacts:** `docker/Dockerfile.mlx`, `scripts/mlx/*`, `patches-mlx/README.md`,
  `.dockerignore`.
- **Next:** build on the Spark, capture the base-image digest, run `tensorfold-dsv41 --version`,
  inspect the real checkpoint, and translate its config.

### 2026-11-09 - Planning only

- **Phase:** pre-Phase 0.
- **Done:** recorded blockers in `Agents/Blockers.md`; wrote the plan as `Agents.md`; created
  this log. No engine, container, or model code changed.
- **Approach used:** source review of the pinned fork (`fbbc971`), upstream TensorFold `main`
  and v1.2.5, the target model card/config/index, and the repo's Dockerfile/patch set.
- **Result:** exact MLX checkpoint is not served by any current TensorFold build. The gap is
  format and architecture admission, not CUDA version.
- **Artifacts:** `Agents/Blockers.md`, `Agents.md`, `Agents/Implementation.md`.
- **Next:** Phase 0. Pin the CUDA base digest and TensorFold SHA, create
  `docker/Dockerfile.mlx`, build on the Spark, and run a tiny fixture.

## Checkpoint layout facts

Read from the published shard headers on 2026-11-09 (no weights downloaded):

| Tensor | Dtype | Shape | Meaning |
|---|---|---|---|
| `layers.0.ffn.experts.w1.weight` | U32 | `[192, 2304, 320]` | 192 experts, N=2304, K=5120 at 2-bit/group 64 |
| `layers.0.ffn.experts.w1.scales` | BF16 | `[192, 2304, 80]` | K/64 = 80 groups |
| `layers.0.ffn.experts.w1.biases` | BF16 | `[192, 2304, 80]` | same grid |
| `layers.0.ffn.shared_experts.w1.weight` | U32 | `[2304, 960]` | 6-bit/group 64, K=5120 |
| `layers.1.engram.embed.weight` | U32 | `[384006168, 32]` | 4-bit/group 32, K=256 |
| `layers.1.engram.embed.scales` | BF16 | `[384006168, 8]` | K/32 = 8 groups |
| `layers.1.engram.wkv.weight` | U32 | `[25600, 1152]` | 6-bit/group 64, K=6144 |
| `layers.0.attn.wo_a.weight` | BF16 | `[8192, 4096]` | 8 groups of 1024 rows; unquantized |

The packed-word rule is `packed_last = K * bits / 32`; scales/biases are `K / group`.
The expert tensors are stacked on the expert axis, so the loader can either gather from
`[192, N, packedK]` or split once at load. The target's `wo_a` is one `[8192, 4096]` tensor;
the EXL3 loader expects `attn.wo_a.slice.N`, so the affine path must handle the whole tensor
or split it.

## Environment and build records

Fill one row per image built. Keep the digest, not just the tag.

| Date | Host | Arch | Driver | Base image | Base digest | CUDA | TensorFold ref | Zig | Image digest | Command |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | local Windows (syntax only) | x86_64 | - | nvcr.io/nvidia/pytorch:26.07-py3 (tag) | - | - | 26e1b074... | 0.17.0 | - | not built |
| - | DGX Spark (GB10) | arm64 | - | nvcr.io/nvidia/pytorch:26.07-py3 | - | - | 26e1b074... | 0.17.0 | - | pending |

## Actual approaches used

Record the approach that was actually implemented, even if it differs from `Agents.md`.

| Area | Planned approach | Actual approach | Why it changed | Result |
|---|---|---|---|---|
| Base engine | Upstream `main` first, fork fallback | Upstream `main` at `26e1b074...` | Has affine readers and the dsv41 lane backend | Dockerfile targets it; build pending |
| Config | Translate `omlx_deepseek_v41.quantized_modules` to canonical metadata | Direct canonical per-module keys inside `quantization`; non-affine modes emitted as `false` plus `non_affine_modules` | Reuses `qwen27/affine.zig` parser and keeps MXFP8/MXFP4 visible to the dsv41 patch | Synthetic fixture and published config pass (429 affine, 31 non-affine); `0001` config gate committed |
| Affine layout | Validate packed U32 shapes before reading | Host-only `affine.zig` parser and shape checks | Starts the loader without touching CUDA kernels | `0002` committed; 2 unit tests pass |
| Affine plan | Enumerate modules from headers | Header-only `affine_plan.zig`; dense/non-affine/Engram listed as skipped | Keeps the first binding pass free of payload reads | `0004` committed; 15-test host suite passes |
| Boot admission | Wire the plan into `cuda_model` and load weights | Plan + header validation run at boot, then `AffineForwardNotPorted` | Avoids a half-wired forward and keeps the EXL3 path untouched | `0005` committed |
| 192 experts / world 1 | Data-driven expert count; single-rank path | `affine_plan` exercises 192; `block_admission` test proves world 1 | Plan already uses `cfg.experts`; no hardcoded 384 in the affine path | `0006` committed |
| Mixed DSpark | Load MXFP8/MXFP4 modules | Plan lists them in `mixed`; boot logs the count | Keeps affine and mixed formats separate until readers exist | `0007` committed; 17 tests |
| Affine reference | Verify dequantization math before kernels | CPU bit-stream extraction and row dot | Gives kernels an independent oracle, including 6-bit word crossings | `0008` committed; 3 tests |
| CUDA width audit | Map formats to native kernels | Linears use `affine8.cu` 2/3/5/6/8-bit; experts only 3/4-bit native | Target's 2-bit experts require a per-expert fallback | `0009` committed; 18 tests |
| Expert fallback | Call `AffineN` per active expert | Byte views into the stacked projection with bounds checks | Bridges the missing 2-bit expert kernel without a new CUDA kernel | `0010` committed; 19 tests |
| Mixed reader | Load DSpark MXFP8/MXFP4 | U32 bit-stream words with U8 E8M0 scales, no bias | The mixed modules reuse the same code stream as affine | `0011` committed; 35 tests |
| Forward dispatch | Map modules to kernels | `.affine8_linear`, `.affine_n_expert`, `.mxfp8`, `.mxfp4` table | Separates format selection from buffer binding | `0012` committed; 18 tests |
| Binding layouts | Freeze shapes for device binding | `affine_exec.zig` carries n/k/bits/group per entry | Device allocation and launch remain separate | `0013` committed; 25 tests |
| Device bindings | Resolve loaded tensors to addresses | `affine_bind.zig` checks bytes and freezes ptr/len | Verifies the loader and binding contract before launch | `0014` committed; 26 tests |
| One-layer fixture | Test one layer without 211 GiB load | `make_one_layer.py` + layer-selection patch | Shortens the 1o/1p loop and memory footprint | `0015` committed; 48 tensors |
| AffineN packing | Device tiles for the launch | `affine_pack.zig` allocates and packs word/scale/bias buffers | First device-side affine step | `0016` committed; stub-CUDA test |
| AffineN launch | Run a projection | `affine_launch.zig` allocates x/sums/out and calls decode/prompt | First actual affine kernel call | `0017` committed; stub-CUDA test |
| Affine smoke | Compare one projection to golden | `affine_smoke.zig` loads `wq_a`, packs/decodes two rows | Proves reader→pack→launch before the full layer | `0018` committed; GPU run pending |
| Engram | Convert 4-bit affine tables to engine rows | FP8 E4M3/E8M0 row shards (`264 B/row`) | Matches `engram_rows.zig` without engine changes | Converter + unit checks pass; ~203 GB output |
| Expert layout | Prefer native stacked gather; else load-time split | Raw reader preserves the stacked `[E, N, packed]` tensor; no split yet | Target has one tensor per projection | `0003` committed |
| Affine kernels | Reuse upstream affine readers, add dsv41 expert path | Not started | - | - |
| Engram | Offline convert to `TF_DS_ENGRAM` first | Inspection only | - | Not started |
| TP/world | World 1 serving path | Not started | - | - |
| Context | 32K-64K first | Not started | - | - |

## Test and measurement log

| Date | Test | Command | Result | Notes |
|---|---|---|---|---|
| 2026-11-09 | Python syntax | `python -m py_compile scripts/mlx/*.py` | Pass | Local Windows |
| 2026-11-09 | Shell syntax | `bash -n scripts/mlx/run_container.sh` | Pass | Git Bash |
| 2026-11-09 | `0001` patch apply | `git apply --check patches-mlx/0001-admit-omlx-affine.patch` | Pass | Against pinned `26e1b074` |
| 2026-11-09 | `0002` patch apply | `git apply` in a fresh worktree with `0001` | Pass | Both patches apply |
| 2026-11-09 | `0003` patch apply | `git apply` on `0001`+`0002` | Pass | New reader + `named`/`plan` hunks |
| 2026-11-09 | `0004` patch apply | `git apply` on `0001`-`0003` | Pass | Plan + `dense_by_default` |
| 2026-11-09 | `0005` patch apply | `git apply` on `0001`-`0004` | Pass | Boot guard |
| 2026-11-09 | `0005` syntax | `zig ast-check` on `cuda_model`/`affine_load` | Pass | Guard compiles syntactically |
| 2026-11-09 | `0006` patch apply | `git apply` on `0001`-`0005` | Pass | 192-expert/world-1 tests |
| 2026-11-09 | `0007` patch apply | `git apply` on `0001`-`0006` | Pass | Mixed DSpark planning |
| 2026-11-09 | `0008` patch apply | `git apply` on `0001`-`0007` | Pass | Affine reference |
| 2026-11-09 | `0009` patch apply | `git apply` on `0001`-`0008` | Pass | CUDA width audit |
| 2026-11-09 | `0010` patch apply | `git apply` on `0001`-`0009` | Pass | Per-expert fallback views |
| 2026-11-09 | `0011` patch apply | `git apply` on `0001`-`0010` | Pass | MXFP8/MXFP4 reader |
| 2026-11-09 | `0012` patch apply | `git apply` on `0001`-`0011` | Pass | Forward dispatch table |
| 2026-11-09 | Dispatch table | `zig test` on `affine_dispatch.zig` | Pass | 18-test host suite |
| 2026-11-10 | Stage 1 scripts syntax | `python -m py_compile` + `bash -n` | Pass | plan/reference/engram/orchestrator |
| 2026-11-10 | Dispatch dry-run | `plan_dispatch.py` on published config | Pass | 309 affine8 / 120 fallback / 22 MXFP8 / 9 MXFP4 |
| 2026-11-10 | `0013` patch apply | `git apply` on `0001`-`0012` | Pass | Binding layouts |
| 2026-11-10 | Binding layouts | `zig test` on `affine_exec.zig` | Pass | 25-test host suite |
| 2026-11-10 | `0014` patch apply | `git apply` on `0001`-`0013` | Pass | Device bindings |
| 2026-11-10 | Device bindings | `zig test` on `affine_bind.zig` | Pass | 26-test host suite |
| 2026-11-10 | `0015` patch apply | `git apply` on `0001`-`0014` | Pass | One-layer selection |
| 2026-11-10 | One-layer fixture list | `make_one_layer.py --list` | Pass | 48 tensors for layer 0 |
| 2026-11-10 | Memory estimate | `memory_estimate.py` | Pass | 32K ≈112.7 GiB, 1M ≈134.8 GiB |
| 2026-11-10 | Engram converter math | `test_convert_engram.py` | Pass | E4M3/E8M0 roundtrip rms 0.0793 |
| 2026-11-10 | Engram converter fixture | `convert_engram.py --limit-rows 4 --verify` | Pass | 5152-byte shard; max/RMS 0 |
| 2026-11-10 | `0016` patch apply | `git apply` on `0001`-`0015` | Pass | AffineN packing |
| 2026-11-10 | AffineN packing | `zig test` on `affine_pack.zig` with stub CUDA | Pass | Target tile sizes non-zero |
| 2026-11-10 | `0017` patch apply | `git apply` on `0001`-`0016` | Pass | AffineN launch |
| 2026-11-10 | AffineN launch | stub-CUDA test of `affine_launch.linear` | Pass | Allocates x/sums/output |
| 2026-11-10 | `0018` patch apply | `git apply` on `0001`-`0017` | Pass | Affine smoke |
| 2026-11-10 | Affine smoke syntax | `zig ast-check` on `affine_smoke`/`cuda_model`/`named` | Pass | GPU run pending |
| 2026-11-09 | Mixed reader | `zig test` on `mixed`/`mixed_load` | Pass | 35-test host suite |
| 2026-11-09 | Expert fallback views | `zig test` on `affine_cuda.zig` | Pass | 19-test host suite |
| 2026-11-09 | CUDA width audit | `zig test` on `affine_cuda.zig` | Pass | 18-test host suite |
| 2026-11-09 | Affine reference | `zig test` on `affine_ref.zig` | Pass | 2/6/8-bit, 3 tests |
| 2026-11-09 | Mixed DSpark planning | `zig test` on `affine_plan.zig` | Pass | 17-test host suite |
| 2026-11-09 | 192-expert plan | `zig test` on `affine_plan.zig` | Pass | 16-test host suite |
| 2026-11-09 | World-1 admission | `zig test` on `block_admission.zig` | Pass | 9 tests |
| 2026-11-09 | Affine plan enumeration | `zig test` on `affine_plan.zig` | Pass | 15-test host suite |
| 2026-11-09 | Affine reader syntax | `zig ast-check` on changed files | Pass | 5 files |
| 2026-11-09 | Affine parser/layout | `zig test` on `affine.zig` | Pass | 2 unit tests; Zig 0.17.0 |
| 2026-11-09 | Config admission | `zig test` on `config.zig` | Pass | 8 tests; fixed affine expert-count path |
| 2026-11-09 | Synthetic fixture | `inspect_checkpoint.py` + `translate_config.py` | Pass | 5 modules; stacked experts and Engram detected; missing w2/w3 intentional |
| 2026-11-09 | Real config translation | `translate_config.py --check` on published config/index | Pass | 460 modules: 429 affine, 22 MXFP8, 9 MXFP4; all pieces present |
| 2026-11-09 | Real index inspection | `inspect_checkpoint.py` on published config/index | Pass | 21 shards, 2286 tensors, 211.4 GiB; 378 stacked expert tensors; 16 Engram tensors |

## Open issues and deviations

- Stage 1a-1n plus prep admit metadata, read bytes, enumerate the plan, validate it at boot, exercise 192 experts/world 1, list mixed DSpark modules, provide CPU dequantization oracles, audit CUDA widths, define per-expert fallback views, read mixed tensors, build the dispatch table, freeze binding layouts, resolve device addresses, build a one-layer fixture, convert Engram to FP8, pack and launch `AffineN`, and add a one-projection smoke test. The next gate is the Spark run of `SMOKE=1`; the per-expert loop and full forward remain.
- Engram conversion is ready but expands to ~203 GB and requantizes to E4M3; a native 4-bit reader would avoid both, at the cost of engine work. The `engram.json` token map is still required.
- Phase 0 harness is committed; the image build is still pending on the Spark.
- Upstream `main` is chosen for Phase 0; fall back to the pinned fork if `dsv41-serve` fails to build or admit a fixture.
- The DSpark head is 22 MXFP8 + 9 MXFP4 modules, not affine; mixed-format loading is an additional Phase 1 task.
- Runtime assets: `TF_DSV41_ASSETS` expects RoPE tables and `engram.json`; the target repo does not ship them. The fork's kit may be reusable, but this must be verified before a full-model load.
- Docker is not installed on the local Windows host; the build must run on the Spark or a Linux host with the NVIDIA container runtime.
- The base-image digest is not captured yet.
- The `dsv41-serve` build command and the upstream dsv41 family are unverified.

## Next actions

1. On the Spark: build the image, create the one-layer fixture, and run `CONTAINER=1 SMOKE=1 test_stage1.sh`.
2. Compare the smoke `y[0]` values with `reference_layer.json`; record the error in this file.
3. Wire the per-expert fallback loop and decide the MXFP4 path.
4. Then move to the full model and server qualification.
