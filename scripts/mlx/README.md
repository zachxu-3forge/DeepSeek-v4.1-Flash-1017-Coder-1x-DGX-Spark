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
- `OUT_DIR` — output directory, mounted at `/out`.
- `BUILD=1` — build the image first.
- `DRY_RUN=1` — print the `docker run` command without running it.

The image builds upstream TensorFold (`TF_REF` in `docker/Dockerfile.mlx`) with Zig 0.17.0 and
`sm_121`; no host CUDA or TensorFold is used. The host needs Docker and the NVIDIA container
runtime only.

## Phase 0 gate

The harness is ready when the image builds, `tensorfold-dsv41 --version` runs, and the engine
admits a 192-expert, world-1 fixture with affine and mixed-format modules. The last step needs
the first `patches-mlx` change; until then the gate stays open.
