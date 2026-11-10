# scripts/mlx

Phase 0 tools for the MLX DeepSeek-V4.1 checkpoint. They run on the host or inside the
container and need only Python's standard library.

## Checkpoint inspection

```sh
python3 scripts/mlx/inspect_checkpoint.py "$MODEL_DIR" --headers --json out/inspect.json
```

Expected on the published checkpoint:

- 21 shards, 2,286 tensors, 211.4 GiB.
- 460 quantized modules: 429 affine (2/6/8-bit), 22 MXFP8, 9 MXFP4.
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

`test_stage1.sh` runs the inspector, translator, dispatch counter, CPU golden rows, Engram
checks, and optionally the container boot. Results land in `out/stage1/<timestamp>/` with a
`summary.md`. The detailed acceptance criteria are in `stage1_tests.md`.

## One-layer fixture and memory

```sh
python3 scripts/mlx/make_one_layer.py "$MODEL_DIR" /models/mlx-one-layer --layer 0
python3 scripts/mlx/memory_estimate.py "$MODEL_DIR"
```

`make_one_layer.py` copies layer 0 plus the embedding, final norm, and head into a translated
fixture for `TF_DSV41_LAYERS=0-0`. `memory_estimate.py` reports weights + conservative KV +
reserve against 128 GiB at each context.

## Engram conversion

```sh
python3 scripts/mlx/convert_engram.py "$MODEL_DIR" /engram-converted --dry-run
python3 scripts/mlx/convert_engram.py "$MODEL_DIR" /engram-converted --limit-rows 64 --verify
```

Writes `engram-l<layer>-r0of1.bin` in the engine's FP8/E8M0 row format. The full output
is ~203 GB; the conversion dequantizes the target's 4-bit groups and requantizes to E4M3,
so it is slightly lossy. The `engram.json` token map/kit is still required. Use
`--limit-rows` first and `--verify` to see max/RMS error.

## Phase 0 gate

The harness is ready when the image builds, `tensorfold-dsv41 --version` runs, and the engine
admits a 192-expert, world-1 fixture with affine and mixed-format modules. The last step needs
the first `patches-mlx` change; until then the gate stays open.
