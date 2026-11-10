# Stage 1 test harness

These scripts are meant to be run on the Spark (or any Linux host with the checkpoint)
before and after stages 1m/1n/1o. They write a permanent result directory so the
deployment can be reviewed later.

## Prerequisites

- `MODEL_DIR` points at the downloaded checkpoint. `config.json` and the index are
  enough for inspection/translation/dispatch; the shards are needed for
  `reference_layer.py` and `test_engram.py`.
- For the container check: the image is built, the NVIDIA container runtime is
  available, and the model is readable.
- `TF_DSV41_ASSETS` (or `ASSETS_HOST`) points at the runtime assets once known.

## Host-only

```sh
MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh --host-only
```

Runs:

- `inspect_checkpoint.py`: shard/tensor/expert/Engram structure.
- `translate_config.py --check`: every quantized module has its tensors.
- `plan_dispatch.py`: expected dispatch counts (307 affine8 linears, 120 expert
  fallbacks, 21 MXFP8, 9 MXFP4). Two affine Engram tables and the DSpark `main_proj`
  MXFP8 are reported separately because they are not affine-plan modules.
- `reference_layer.py`: golden dots for one 8-bit, 6-bit, and 2-bit module, plus one
  MXFP8 and one MXFP4 module.
- `test_engram.py`: golden dots and row counts for the two Engram tables.

## Container

```sh
CONTAINER=1 BUILD=1 MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh
CONTAINER=1 SMOKE=1 MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh
```

Adds a `tensorfold-dsv41` boot with `TF_EXPERIMENTAL=deepseek`, `TF_TP_WORLD=1`, and
`TF_DSV41_DRAFTS=0`, captures the log, and compares its `dispatch:` line with
`plan_dispatch.py`. The guard returns before any weight upload, so it plans all layers and
must report the affine-plan counts (307/120/21/9). The `SMOKE=1` run then loads only layer 0
(`TF_DSV41_LAYERS=0-0`).

The engine needs `quantization_config` in `config.json`. If the checkpoint only has the
OMLX map, point `MODEL_DIR` at a translated overlay rather than editing the source:

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

Use hardlinks (or copies), not symlinks: a symlink to the host path is broken inside the
container, where only the overlay is mounted at `/model`. The engine reads the tokenizer at
startup, so the overlay must include it.

**Before stage 1m** the engine is expected to end with `AffineForwardNotPorted`; the
harness treats that as PASS. **After stage 1m** run with `EXPECT_FORWARD=1`, which
requires the guard to be gone.

## One-layer acceptance (after 1m/1n)

The GPU harness must reproduce `reference_layer.json`:

- the same module/row/expert,
- the same `dot_ones` and `dot_alt` within the documented tolerance,
- the same dispatch counts as `plan_dispatch.json`.

`SMOKE=1` adds a one-projection run: `smoke.log` must contain
`affine smoke layers.0.attn.wq_a: y[0] ones=... alt=...`; compare those two values with
the `language_model.layers.0.attn.wq_a` entry in `reference_layer.json`.

The Engram reader (1o) must reproduce `engram.json`, including the two row counts.
For the FP8 conversion path, `convert_engram.py --limit-rows 64 --verify` must report
bounded max/RMS error, and a full `--dry-run` must confirm the ~203 GB output budget.

## Results

`OUT_DIR` defaults to `out/stage1/<UTC timestamp>/` and contains:

- `summary.md` — PASS/FAIL table,
- `inspect.json`, `quantization.json`, `config.translated.json`,
- `plan_dispatch.json`, `reference_layer.json`, `engram.json`,
- `engine.log` when the container check runs.

Keep the `summary.md` path with the deployment record.
