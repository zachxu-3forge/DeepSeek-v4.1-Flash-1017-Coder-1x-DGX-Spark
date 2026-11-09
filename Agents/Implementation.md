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
| Overall status | Stage 1i committed (CUDA width audit); image build pending |
| Base engine | Upstream `main` at `26e1b074b1512d645698eba269ccfb8ebc974c10` for Phase 0; fork fallback |
| CUDA base image | `nvcr.io/nvidia/pytorch:26.07-py3` (tag; digest pending) |
| TensorFold ref | `26e1b074b1512d645698eba269ccfb8ebc974c10` |
| Zig version | 0.17.0 |
| Model revision | `mbrown-uno/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-MLX` at `c2afc6bb446f612ec0791e03834cdf4f8cb05583` (recorded 2026-11-09) |
| Container image | `tf-dsv41-mlx:dev` (not built) |
| Last updated | 2026-11-09 |

## Phase tracker

| Phase | Status | Gate | Evidence |
|---|---|---|---|
| 0. Harness and admission spike | In progress | Translated config + 192-expert, world-1 affine fixture loads | Harness committed; image build pending |
| 1. Affine loader and layout adapter | In progress | One full layer matches dequantized reference | Stage 1i CUDA width audit committed; 2-bit expert fallback needed |
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

## Work log

Newest entries first. Each entry: date, phase, what was done, command, result, artifacts,
next step.

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

- Stage 1a-1i admit metadata, read bytes, enumerate the plan, validate it at boot, exercise 192 experts/world 1, list mixed DSpark modules, provide the CPU dequantization oracle, and audit CUDA widths. The critical gap is now explicit: linears are served by `affine8.cu` at 2/3/5/6/8-bit, but native expert kernels cover only affine-3/4, so the target's 2-bit routed experts need a per-expert fallback. The affine path deliberately returns `AffineForwardNotPorted`.
- Phase 0 harness is committed; the image build is still pending on the Spark.
- Upstream `main` is chosen for Phase 0; fall back to the pinned fork if `dsv41-serve` fails to build or admit a fixture.
- The DSpark head is 22 MXFP8 + 9 MXFP4 modules, not affine; mixed-format loading is an additional Phase 1 task.
- Runtime assets: `TF_DSV41_ASSETS` expects RoPE tables and `engram.json`; the target repo does not ship them. The fork's kit may be reusable, but this must be verified before a full-model load.
- Docker is not installed on the local Windows host; the build must run on the Spark or a Linux host with the NVIDIA container runtime.
- The base-image digest is not captured yet.
- The `dsv41-serve` build command and the upstream dsv41 family are unverified.

## Next actions

1. Implement stage 1j: the per-expert `AffineN` fallback for the 2-bit routed experts.
2. Implement stage 1k: the MXFP8/MXFP4 DSpark reader.
3. Implement stage 1l: forward mapping and the one-layer harness; keep runtime verification deferred.
4. At the end, build the container, run the one-layer reference, and record all results here.
