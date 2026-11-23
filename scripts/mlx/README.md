# scripts/mlx

Phase 0 tools for the MLX DeepSeek-V4.1 checkpoint. They run on the host or inside the
container and need only Python's standard library.

## Checkpoint inspection

```sh
python3 scripts/mlx/inspect_checkpoint.py "$MODEL_DIR" --headers --json out/inspect.json
```

Expected on the published checkpoint:

- 21 shards, 2,286 tensors, 211.4 GiB.
- 460 quantized modules: 429 affine (2/6/8-bit), 22 MXFP8, 9 MXFP4. The affine plan owns
  307 linears and 120 2-bit expert stacks; `convert_dspark_affine.py` moves the 31 DSpark
  modules into affine, giving 331 linears and 129 expert stacks. Two affine Engram tables
  and the DSpark `main_proj` are handled separately.
- 378 stacked expert tensors (one tensor per projection, not per expert).
- 16 Engram tensors.

`--headers` reads safetensors headers and prints packed-affine shape checks; it does not load
weights.

## Config translation

```sh
python3 scripts/mlx/translate_config.py "$MODEL_DIR" \
  --check --out out/quantization.json --write-config out/config.translated.json
```

The translator emits canonical per-module keys under `quantization`, marks non-affine modules
`false`, and records them under `non_affine_modules`. It never modifies the source checkpoint.

`--check` verifies that every quantized module has `.weight` and the expected metadata
(`.scales`/`.biases` for affine, `.scales` for MXFP8/MXFP4). It passes on the published
config/index.

## Translated-config overlay

The engine reads `quantization_config`; the published checkpoint only has the OMLX map.
Build a separate directory and leave the source untouched:

```sh
OVERLAY=/models/mlx-translated
mkdir -p "$OVERLAY"
for f in /models/mlx/model-*.safetensors; do
  ln "$f" "$OVERLAY/$(basename "$f")" 2>/dev/null || cp "$f" "$OVERLAY/"
done
cp /models/mlx/model.safetensors.index.json "$OVERLAY/"
for f in tokenizer.json tokenizer_config.json chat_template.jinja special_tokens_map.json generation_config.json; do
  [ -f "/models/mlx/$f" ] && cp -p "/models/mlx/$f" "$OVERLAY/"
done
python3 scripts/mlx/translate_config.py /models/mlx --write-config "$OVERLAY/config.json"
```

Hardlinks cost no extra disk. Do not use symlinks to the source: inside the container only
the overlay is mounted at `/model`, so an absolute host-path symlink dangles. The engine also
reads the tokenizer at startup, so the overlay needs those files.

## Container

```sh
BUILD=1 scripts/mlx/run_container.sh tensorfold-dsv41 --version
```

Environment:

- `MODEL_DIR` — checkpoint directory, mounted at `/model`.
- `ENGRAM_DIR` — Engram directory, mounted at `/engram`; defaults to `MODEL_DIR`.
- `HF_CACHE` — Hugging Face cache, mounted at `/hf`.
- `ASSETS_HOST` — optional runtime assets directory (RoPE tables, `engram.json`), mounted at
  `/assets`; `TF_DSV41_ASSETS` points there when set.
- `OUT_DIR` - output directory, mounted at `/out`.
- `ENGRAM_SHARDS` - packed Engram shards for `test_stage1.sh`'s `test_engram.py --shard-dir`.
- `BUILD=1` - build the image first.
- `DRY_RUN=1` - print the `docker run` command without running it.
- `TF_EXPERIMENTAL=deepseek` - required by the engine's gated DeepSeek-V4.1 family;
  `test_stage1.sh` sets it for the guard and smoke runs.
- `TF_TP_WORLD=1` - single-Spark run. The engine defaults to world 2 and waits 300 s
  for a peer otherwise; `test_stage1.sh` sets it for the container runs.

The image builds upstream TensorFold (`TF_REF` in `docker/Dockerfile.mlx`) with Zig 0.17.0 and
`sm_121`; no host CUDA or TensorFold is used. The host needs Docker and the NVIDIA container
runtime only.

## Stage 1 verification

```sh
MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh --host-only
CONTAINER=1 BUILD=1 MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh
CONTAINER=1 EXPECT_FORWARD=1 MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh
CONTAINER=1 SMOKE=1 MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh
```

`test_stage1.sh` runs the inspector, the one-layer fixture byte check, the translator and
dispatch checks, CPU golden rows, Engram checks, and optionally the container boot. Results
land in `out/stage1/<timestamp>/` with a `summary.md`. The exact GPU-window sequence is in
`scripts/mlx/stage1_gpu_run.md`. The detailed acceptance criteria are in `stage1_tests.md`.

## One-layer fixture and memory

```sh
python3 scripts/mlx/make_one_layer.py "$MODEL_DIR" /models/mlx-one-layer --layer 0
python3 scripts/mlx/memory_estimate.py "$MODEL_DIR"
```

`make_one_layer.py` copies layer 0 plus the embedding, final norm, and head into a translated
fixture for `TF_DSV41_LAYERS=0-0`. `memory_estimate.py` now follows the engine's KV layout:
SWA layers keep `2*window`, compressed values/scales are stored once per KV source, index
keys once per index source, plus the RoPE tables. At 600K it reports a **1.58 GiB** KV, 108.8
GiB total and 19.2 GiB headroom; the live 600K boot measured 99 GiB used / 22 GiB available.
1M is ~110 GiB on paper but remains out of scope.

## RoPE tables

```sh
python3 scripts/mlx/make_rope_tables.py /assets/rope.bin
```
Writes the dsv41 CUDA `rope.bin`: 16-byte header, then the plain window table and the
YaRN compressed table, each f32 `[rows, 64]`. The default `--rows 602048` covers 600K plus a
2,048-row prefill segment. The compressed table is **unscaled** by default, matching
`omlx/patches/deepseek_v41/language.py`; `--mscale` opts into the MiaAI/Qwen convention
that bakes `mscale = 0.1 ln(factor) + 1` into the table. Verify the frequencies against
`core/rope_yarn.zig` and the target's MLX reference before serving. `--self-test` checks
the known Qwen3.6 YaRN vector; `--check FILE` validates an existing asset's header, size,
sampled cos/sin values, and scale (`--check FILE --mscale` for the scaled form).

Locally generated: `602048` rows, `308,248,592` bytes, `--check` passes.

## Boot assets

`cuda_model.boot` reads three things under `TF_DSV41_ASSETS` before the runner starts:
`rope.bin`, `engram-token-map.bin`, and `aot/aot.json` (the affine path launches no Triton
kernels, so an empty set is enough).

```sh
mkdir -p /assets/aot
printf '{"kernels":[]}\n' > /assets/aot/aot.json
docker run --rm -v "$MODEL_DIR:/model:ro" -v /assets:/assets "$IMAGE" \
  tf-dsv41-token-map --tokenizer /model/tokenizer.json --config /model/config.json \
  --output /assets/engram-token-map.bin
```

The map builder checks the release tokenizer's compressed vocabulary count (99,092 keys)
and writes `engram-token-map.bin.json` beside it. The engine reads the map, not the
manifest; use the translated config the engine will boot with.

## Triton AOT set

The affine decode path still launches the family's Triton kernels. Without their cubins the
engine admits the model but refuses each request with `MissingTritonVariant`. The set is
generated from the recipe's pinned Python reference tree; the image carries the required
Triton 3.7.1 and ptxas 13.3.73.

```sh
git clone --depth 1 https://github.com/jayleaton/deepseek-v41-tensorfold-spark recipe
(cd recipe && scripts/fetch-reference.sh build/reference)      # build/reference/src
IMAGE=tf-dsv41-mlx:dev
R=/home/deploy/repos/deepseek-v41-tensorfold-spark
A=/home/deploy/repos/DeepSeek-v4.1-Flash-Coder-1017-1x-DGX-Spark/checkpoints/assets
PY="docker run --rm --user $(id -u):$(id -g) -v $R:/recipe -v $R/build/reference/src:/refsrc:ro -v $A:/assets -w /recipe/engine/zig -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/triton --entrypoint python3 $IMAGE"
$PY tools/zig/triton_fill.py sigs --py /refsrc --out /assets/sigs.json
docker run --rm --user $(id -u):$(id -g) -v /path/model:/model:ro -v $A:/assets \
  -e TF_EXPERIMENTAL=deepseek -e TF_TP_WORLD=1 -e TF_DSV41_DRAFTS=0 -e TF_DSV41_GRAPHS=0 \
  -e TF_DSV41_CONTEXT=4096 -e TF_DSV41_AOT_CONFIG=/model/config.json \
  --entrypoint /opt/tensorfold/bin/tf-dsv41 $IMAGE aot-needs /assets/sigs.json /assets/needs.json
$PY tools/zig/triton_fill.py compile --py /refsrc --needs /assets/needs.json \
  --options tools/zig/dsv41_triton_options.json --arch 121 --have /assets/aot --out /assets/aot
```

At 4,096 context the affine decode plan needs 27 variants (from 12,545 programs); the fill
reported 27 compiled, 0 failed. Re-run `aot-needs` and `compile --have "$ASSETS/aot"` after
changing context, slots, drafts, or a computation knob. `aot-needs` uses
`TF_DSV41_AOT_CONFIG` so the affine format and its pruned 192-expert count are visible; it
skips native prefill (the affine path runs prompts as decode windows).

## Engram conversion

The chosen path is the engine's native 4-bit reader, streamed from SSD: 160 B/row, ~114 GiB,
no requantization. The converter writes that format by default; `--format fp8` is the
fallback/tool.

```sh
python3 scripts/mlx/convert_engram.py "$MODEL_DIR" /engram-native --dry-run
python3 scripts/mlx/convert_engram.py "$MODEL_DIR" /engram-native --verify
python3 scripts/mlx/test_engram.py "$MODEL_DIR" --shard-dir /engram-native
```

Native rows are the source's own bytes: 32 U32 affine-4 words, then 8 BF16 scales and 8 BF16
biases (group 32). The header records `row_bytes=160`; the reader also accepts the older
FP8/E8M0 format (264 B: 256 E4M3 values and 8 E8M0 scales). `--verify` decodes 64 sampled rows
back from the written file and compares them with the release. The OMLX map lists
`engram.wkv` but omits `engram.embed`; the scripts infer affine 4-bit/group-32 when the
table tensors are present. The `engram.json` token map/kit is still required.

`test_engram.py --shard-dir` checks row 0 of every packed shard against the release bytes
(native) or the decoded values (FP8).

## DSpark conversion

The checkpoint's DSpark blocks are MXFP8/MXFP4, not affine. To keep one kernel family, convert
them once into a separate overlay; the source checkpoint is not modified:

```sh
python3 scripts/mlx/convert_dspark_affine.py "$MODEL_DIR" /model-dspark --verify
```

The 22 MXFP8 linears (attention, shared experts, `main_proj`) become affine-8/group 64; the 9
MXFP4 expert stacks become affine-3/group 64. Affine-3 is deliberate: `AffineN` serves
2/3/5/6/8-bit codes, not 4. The converter writes one `model-dspark-affine.safetensors` shard,
updates the index and config, and hardlinks the source shards. It copies `config.json` and the
index (never hardlinks them), so the source directory stays intact. Run the image with
`--drafts` to use it; `--no-drafts` is still the faster default with the current batched MoE.

## Stage 1 gate

The harness is ready: the image builds, `tensorfold-dsv41 --version` runs, and the engine
admits the 192-expert, world-1 affine plan. The Stage 1p guard logs
`307 affine8 linears, 120 expert fallbacks, 21 mxfp8, 9 mxfp4` and the layer-0 projection
sweep (every 8/6-bit linear and all three 2-bit expert stacks) matches the CPU rows within
BF16 rounding.

Verified on `dgx2` at image `055886260245` (patches `0001`-`0044`): host checks and the
container guard PASS; the projection smoke matches the reference; the layer-0 fixture loads
(dense 17 tensors/2.53 GiB, 7 linears/0.10 GiB, 3 expert stacks/1.98 GiB) and answers
`POST /v1/completions` with HTTP 200 (5-token prompt, 4 tokens, 3 decode rounds). The affine
decode path needs the 27-variant Triton AOT set described above.

At image `0e6645b555ce` (patches `0001`-`0049`) the full 40-layer model loads in 97.7 s
(309 linears/4.28 GiB, 120 expert stacks/79.10 GiB, 83.37 GiB affine; 96 GiB resident with
24 GiB available) and answers a request (0.47 s TTFT, 2.2 tok/s with host-loop experts).
The text is not yet reference-equal: the compressed-layer KV/indexer packing (B4) and the
synthetic Engram rows still differ.

At image `1df195f247ca` (patches `0001`-`0054`) the text is coherent. The missing piece was
the persistent bf16 mHC `fn16` form: the affine boot skipped the `forms` pass, so every
layer's pre/post mixing read uninitialized memory. `0054` builds those forms for affine too.
The native Engram tables are packed and used, and the full model answers `Paris` to the
capital question and the checkpoint's `bench.json` LRU prompt opening. The remaining
divergence from omlx is the bf16 KV cache vs its packed 4-bit/e4m3 formats and the affine
activation quantization.

At image `7fd2b99bf79a` (patches `0001`-`0057`) DSpark runs on the converted overlay: 331
affine linears (4.74 GiB), 129 expert stacks (84.64 GiB), 620 dense tensors (5.51 GiB), and
`drafts: DSpark + copies`. At `5bec44994cf6` (patches `0001`-`0060`) the affine experts are
packed once at load and the MoE launches are batched, so decode is **9.9 tok/s** and TTFT
**1.04 s** on the 64-token LRU prompt (same token SHA `7f8834716518` as the slow path). DSpark
accepts 44/68 and reaches 8.46 tok/s, so the default remains `--no-drafts` until the verify
path is fused. 600K no-drafts: 104 GiB used / 17 GiB available, 9.32 tok/s.

## Layer reference

`reference_affine_layer.py` runs the recipe's PyTorch reference (`engine/reference`) against
the MLX affine checkpoint with a dequantizing loader: U32 words + BF16 scales/biases per
group, stacked experts sliced per expert, and the dense `wo_a` split into its output groups.
It needs the recipe checkout at `/recipe`, the fixture at `/model`, and writes float32
`[T, hc, hidden]` streams and (with `--sublayers`) the attention/MoE outputs:

```sh
python3 scripts/mlx/reference_affine_layer.py 671,6102,294,8760,344 --layers 1 \
  --keep-layer 0 --sublayers --out /out/ref_layer0.npy
```

Layer 0 on the fixture matches the engine's traced streams at correlation `0.9997`; this is
the CPU oracle for the remaining layer and full-model comparisons.
