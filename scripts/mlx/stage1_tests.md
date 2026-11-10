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
- `plan_dispatch.py`: expected dispatch counts (309 affine8 linears, 120 expert
  fallbacks, 22 MXFP8, 9 MXFP4).
- `reference_layer.py`: golden dots for one 8-bit, 6-bit, and 2-bit module, plus one
  MXFP8 and one MXFP4 module.
- `test_engram.py`: golden dots and row counts for the two Engram tables.

## Container

```sh
CONTAINER=1 BUILD=1 MODEL_DIR=/models/mlx scripts/mlx/test_stage1.sh
```

Adds a `tensorfold-dsv41` boot with `TF_DSV41_LAYERS=0-0` and `TF_DSV41_DRAFTS=0`,
captures the log, and compares its `dispatch:` line with `plan_dispatch.py`.

**Before stage 1m** the engine is expected to end with `AffineForwardNotPorted`; the
harness treats that as PASS. **After stage 1m** run with `EXPECT_FORWARD=1`, which
requires the guard to be gone.

## One-layer acceptance (after 1m/1n)

The GPU harness must reproduce `reference_layer.json`:

- the same module/row/expert,
- the same `dot_ones` and `dot_alt` within the documented tolerance,
- the same dispatch counts as `plan_dispatch.json`.

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
