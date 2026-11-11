# Implementation Plan: MLX DeepSeek-V4.1 on one DGX Spark

Status: proposed. Read `Agents/Blockers.md` first; this plan assumes we attempt the exact
`mbrown-uno/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-MLX` checkpoint, not the GGUF fallback.
Implementation progress, decisions, and the approaches actually used are recorded in
`Agents/Implementation.md`; update that file as work proceeds.

## Constraints

- **Docker only.** The host stays untouched: no host CUDA toolkit, driver, TensorFold, Zig, or
  Python install. Everything is built and run in the container.
- **`docker run --rm`.** No state outside mounted volumes. Model, Engram, HF cache, and outputs
  are host directories mounted read-only or read-write as needed.
- **Latest CUDA/TensorFold in the image.** The image pins a CUDA base by digest and a
  TensorFold ref by commit. "Latest" is allowed for experiments, but every milestone records
  the exact digest/SHA; `main` is not a reproducible pin.
- **One GB10, 128 GB.** Start at 32K, then target 600K context with YaRN. 1M context is
  out of scope for the first working milestone.
- **Exact checkpoint.** The MLX weights are the source of truth; no conversion to EXL3 unless
  the port is abandoned.

## Recommended base

Use **upstream TensorFold `main`** as the base, not the pinned fork.

Why: upstream already has the MLX affine readers (2/3/4/5/6/8-bit, groups 32/64/128) and a
DeepSeek-V4.1 family with a lane backend and world-1/3/4 tests. The pinned fork has the proven
GB10/TP2 serving path but is EXL3-only. Porting affine into upstream is less work than porting
the whole dsv41 family into the fork.

Risk: upstream's DeepSeek-V4.1 family is unreleased and PR #827 is closed/unmerged. Phase 0
must prove the family builds and runs a tiny fixture before committing to it. If it does not,
fall back to the fork base and port affine + world 1 into it.

## External reference: MiaAI-Lab Qwen3.8 recipe

`MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold` runs TensorFold v0.6.0's Python/Triton
`qwen3_5` engine, not this Zig/CUDA `deepseek_v41` base. Its patches do not apply; use it as
evidence and a design reference:

- **YaRN:** its `yarn.py` matches `core/rope_yarn.zig` (beta 32/1, `mscale = 0.1 ln(factor) + 1`).
  Use it to cross-check a 600K `rope.bin`, and confirm whether `mscale` is baked into the table
  or applied by the kernel.
- **FP8 KV:** its e4m3 cache halves KV with KL 0.0031 and 98.7% top-1 agreement. The dsv41 CUDA
  path does not wire `kv8`; use bf16 first and port FP8 only if 600K does not fit.
- **Memory:** do not set the reserve to 0 on a shared box; try 4-8 GiB. A pinned pool is a
  policy to copy, not its PyTorch code; check `TF_DSV41_POOL_TOKENS`/`kv_pool` first.
- **Operations:** its `config.sh`/`.env`/start-stop scripts, health smoke test, and measurement
  tables are good patterns for this repo.
- **Do not port:** the image/video patch, `qwen3_5` Triton kernels, DFlash2, Qwen geometry, or
  the `YARN_FACTOR` override. This model already has YaRN factor 16 from 65,536.

## Docker contract

Add `docker/Dockerfile.mlx` (or extend `zig/Dockerfile` behind a build arg) that:

1. Starts from an arm64 CUDA devel/PyTorch image pinned by digest. Default candidate:
   `nvcr.io/nvidia/pytorch:26.07-py3` (known good for `sm_121`), with an override for a newer
   tag. Do not rely on the host's CUDA.
2. Installs Zig at a pinned version (currently 0.17.0) and any build deps.
3. Fetches TensorFold at `TF_REF` (default: tested upstream commit; `main` only for spikes),
   applies `patches-mlx/*.patch`, and builds `tensorfold-dsv41` and `tf-dsv41` for `sm_121`
   with `zig build dsv41-serve`.
4. Copies only the binaries, licenses, and any helper scripts into a small runtime stage.
5. Has no model weights baked in.

Run shape:

```sh
docker build -f docker/Dockerfile.mlx \
  --build-arg BASE_IMAGE=<digest> --build-arg TF_REF=<sha> \
  -t tf-dsv41-mlx:dev .

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/hf" \
  -v "/path/to/model:/model:ro" \
  -v "/path/to/engram:/engram:ro" \
  -v "$PWD/out:/out" \
  tf-dsv41-mlx:dev <command>
```

`--gpus all` needs the NVIDIA container runtime on the host; that is a host prerequisite, not
a host CUDA/TensorFold install. Record the driver version in every run log.

## Phases

### Phase 0 - Harness and admission spike (1-3 days)

Deliverables:

- `docker/Dockerfile.mlx` builds an arm64 image with a pinned CUDA base and TensorFold ref.
- `scripts/mlx/inspect_checkpoint.py`: reads the index and config, prints per-module format,
  shapes, expert counts, and memory estimates.
- `scripts/mlx/translate_config.py`: emits a canonical `quantization_config` (or the engine's
  expected metadata) from `omlx_deepseek_v41.quantized_modules`, without modifying the source
  checkpoint.
- A one-layer or tiny-fixture checkpoint that the engine can open.

Gate: the engine parses the translated config and admits a 192-expert, world-1 fixture with
affine and mixed-format modules. No full-model load yet.

### Phase 1 - Affine loader and layout adapter (1-2 weeks)

Tasks:

- Add MLX affine packed-word/scales/biases loading for dsv41 linears.
- Add stacked-expert loading: split `[192, ...]` tensors into the engine's per-expert layout,
  or teach the engine the stacked layout directly. Prefer the latter if the kernels can gather
  from it; otherwise convert once at load.
- Support mixed modules: affine 2/6/8-bit, MXFP8 (22 DSpark modules), MXFP4 (9 DSpark
  expert modules), bf16/MXFP8 `wo_a`, and unquantized tensors.
- Make expert count, top-k, and DSpark expert count data-driven (192/6, 128/3).
- Add world-1 admission to the serving path.

Gate: one full layer runs a forward pass on GPU and matches a dequantized reference within a
documented tolerance. Compare against a PyTorch/MLX dequant of the same layer, not against the
EXL3 model.

### Phase 2 - Full model, 32K to 600K context (1-2 weeks)

Tasks:

- Load all 40 layers plus DSpark and the vision tower on one rank.
- Implement or adapt 2-bit affine expert gather kernels; verify decode and prefill rows.
- Generate `rope.bin` for 600K plus a prefill segment; verify the YaRN frequencies and
  `mscale` against `core/rope_yarn.zig` before loading.
- Add memory admission for ~83.4 GB affine plus dense/native weights and KV; enforce a
  conservative default context and validate 600K with YaRN. Start with a 4-8 GiB reserve;
  only port dsv41 FP8 KV if bf16 KV does not leave a safe margin.
- Validate greedy generation against omlx on a fixed prompt for 64-256 tokens (accept small
  divergence only where documented).

Gate: `tensorfold-dsv41 serve` answers a text prompt on one Spark; peak device/host memory
stays under the safety floor; 32K context is stable; 600K is measured; no host libraries are
loaded.

### Phase 3 - Engram (1-2 weeks)

Chosen: the native 4-bit reader, streamed from SSD.

- **Native reader:** read the target's U32 4-bit/group-32 rows plus BF16 scales/biases
  (160 bytes/row, ~114 GiB) and dequantize on demand. This avoids the E4M3 requantization
  and is smaller than the FP8 form.
- **Fallback/tool:** `scripts/mlx/convert_engram.py` still writes the engine's FP8 rows
  (`264 B/row`, ~188.83 GiB) for comparison or if the native reader is delayed.

Gate: Engram lookups match the reference for sampled tokens; the native tables stream within
budget; SSD streaming works under `--rm` with a read-only mount.

### Phase 4 - Server, tools, vision (1 week)

Tasks:

- Rebase the portable server/API patches (`0004`, `0027`-`0032`, `0036`, `0046`-`0049`,
  `0057`, `0058`, `0064`-`0067`) onto the chosen base.
- Wire tokenizer and chat template; verify tool calling and thinking on/off.
- Wire the vision tower and image input; test one image request.
- Keep engine/CUDA patches that depend on TP2/RDMA out of the single-rank build unless needed.

Gate: OpenAI-compatible `/v1/chat/completions`, streaming, tools, and one image request work
inside the container.

### Phase 5 - Qualification and documentation (3-5 days)

- Measure TTFT, decode tok/s, peak memory, and context limits on one GB10.
- Run a fixed regression suite (greedy determinism, draft on/off equality where applicable,
  concurrent streams).
- Document the exact image digest, TensorFold SHA, patch set, and run command.
- Record the MiaAI-style tables: decode/prefill by stream count, KV dtype quality, memory
  admission, and what was tuned or deliberately not tuned.
- Record what is still missing for 1M context.

## Repo layout

```
docker/Dockerfile.mlx          # container build, pinned base + TF ref
patches-mlx/                   # engine changes, filename order
scripts/mlx/inspect_checkpoint.py
scripts/mlx/translate_config.py
scripts/mlx/convert_engram.py
scripts/mlx/reference_layer.py # dequantized reference for Phase 1
scripts/mlx/run_container.sh   # docker run --rm wrapper
Agents/Blockers.md
Agents.md                      # this plan
Agents/Implementation.md       # progress and actual approaches
```

## Decision points

1. **Base:** upstream `main` vs pinned fork. Decide after the Phase 0 build/admission spike.
2. **Expert layout:** load-time conversion to per-expert vs native stacked gather. Decide by
   measuring load time and decode speed on one layer.
3. **Engram:** native 4-bit reader chosen (smaller, no requantization); the FP8 converter
   remains a fallback.
4. **Context target:** 32K first, then 600K with YaRN; 1M is out.

## Fallback

If Phase 1 shows the affine path is not tractable in the time box, switch to the source GGUF
with llama.cpp CUDA in the same `--rm` container. That runs the model, but not this engine and
not this repo's patch set. Keep it as a parallel sanity baseline, not a silent pivot.

## First actions

1. Pin the CUDA base digest and the TensorFold SHA; write `docker/Dockerfile.mlx`.
2. Build the image on the Spark and run `tensorfold-dsv41 --version` plus a tiny fixture.
3. Write `inspect_checkpoint.py` and `translate_config.py`.
4. Produce the one-layer affine fixture and get it admitted.
5. Only then start the loader and kernel work.
