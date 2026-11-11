# Stage 1 GPU runbook

Use this only when `dgx1` is idle for this project. Check first:

```sh
nvidia-smi
docker ps --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}'
```

If another workload is using the GPU, do not rebuild or run the engine; wait or coordinate.

## 0. Update and set paths

```sh
cd /home/deploy/Models/DeepSeek-v4.1-Flash-1017/MLX-Coder/Recipe
git pull
export MODEL_DIR=/home/deploy/Models/DeepSeek-v4.1-Flash-1017/MLX-Coder
export WORK=/home/deploy/dsv41-mlx
export OVERLAY=$WORK/model-translated
export BASE_IMAGE="nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
git log -1 --oneline
```

The overlay must contain `config.json` (translated), the index, the tokenizer files, and
hardlinks to every shard. Rebuild it if it is missing.

## 1. Rebuild the image

```sh
BUILD=1 BASE_IMAGE="$BASE_IMAGE" MODEL_DIR="$MODEL_DIR" \
  scripts/mlx/run_container.sh tensorfold-dsv41 --version \
  2>&1 | tee "$WORK/build-stage1q.log"
```

Expected: `tensorfold-native 1.2.5`. Record the image ID:

```sh
docker image inspect tf-dsv41-mlx:dev --format '{{.Id}}'
```

## 2. Host checks (CPU only)

```sh
MODEL_DIR="$MODEL_DIR" OUT_DIR="$WORK/stage1-host-q" \
  scripts/mlx/test_stage1.sh --host-only 2>&1 | tee "$WORK/host-q.log"
```

Expected: all host checks PASS, including `make_one_layer_fixture`,
`test_translate_config`, `test_plan_dispatch`, `test_rope_tables`, and real Engram rows.

For a 600K run, generate the asset once:

```sh
mkdir -p "$WORK/assets"
python3 scripts/mlx/make_rope_tables.py "$WORK/assets/rope.bin"
```

Then pass `ASSETS_HOST="$WORK/assets"`, set `TF_DSV41_CONTEXT=600000`, and start with
`TENSORFOLD_MEMORY_RESERVE_GIB=4..8`. Do not set the reserve to 0 on a shared box.

Pack the native Engram shards once (CPU and disk only; ~114 GiB):

```sh
python3 scripts/mlx/convert_engram.py "$MODEL_DIR" "$WORK/engram-native" --limit-rows 64 --verify
python3 scripts/mlx/convert_engram.py "$MODEL_DIR" "$WORK/engram-native" --overwrite --verify
python3 scripts/mlx/test_engram.py "$MODEL_DIR" --shard-dir "$WORK/engram-native"
```

Then set `ENGRAM_SHARDS="$WORK/engram-native"` for the host checks and
`ENGRAM_DIR="$WORK/engram-native"` for the container. `--verify` decodes sampled rows back
from the written files; native rows must report max 0.

## 3. Guard and layer-0 sweep

```sh
CONTAINER=1 MODEL_DIR="$OVERLAY" OUT_DIR="$WORK/stage1-q" \
  scripts/mlx/test_stage1.sh 2>&1 | tee "$WORK/container-q.log"
```

Expected guard line:

```text
dsv41 affine bytes: linears raw 4.27 GiB, packed 4.28 GiB; experts raw 79.10 GiB (largest packed expert 3.52 MiB); total raw 83.37 GiB, resident with packed linears 83.38 GiB
dsv41 affine: 43 layers, ... dispatch: 307 affine8 linears, 120 expert fallbacks, 21 mxfp8, 9 mxfp4
AffineForwardNotPorted
```

Then the smoke/sweep run:

```sh
CONTAINER=1 SMOKE=1 MODEL_DIR="$OVERLAY" OUT_DIR="$WORK/stage1-q-smoke" \
  scripts/mlx/test_stage1.sh 2>&1 | tee "$WORK/smoke-q.log"
```

Expected in `smoke.log`:

- one `affine smoke cpu <module>: row0 ...` and one `affine smoke <module>: y[0] ...`
  line for every layer-0 linear and all three expert stacks;
- `affine smoke dev layers.0.attn.wq_a: y[0] ones=...` matching the host `wq_a` line;
- `affine smoke attn layers.0: qa0=... q0=... kv0=... kvn0=...` for the attention chain;
- `smoke values match`;
- `dispatch counts match`;
- `AffineForwardNotPorted` after the smoke.

The harness compares every CPU/GPU pair with `abs(gpu - cpu) <= 0.01 + 0.01 * abs(cpu)`
and checks `wq_a`, `shared_experts.w1`, and expert 0 against `reference_layer.json`.

## 4. Resident affine linears and dense weights

When the GPU is idle and the sweep is clean, verify the full packed-linear set and the
dense/native half of the affine checkpoint:

```sh
TF_EXPERIMENTAL=deepseek TF_TP_WORLD=1 TF_DSV41_DRAFTS=0 TF_DSV41_AFFINE_LOAD=1 \
MODEL_DIR="$OVERLAY" OUT_DIR="$WORK/stage1-load" \
  scripts/mlx/run_container.sh tensorfold-dsv41 serve /model \
  2>&1 | tee "$WORK/affine-load.log"
```

Expected:

```text
dsv41 affine bytes: linears raw 4.27 GiB, packed 4.28 GiB; ...
dsv41 affine dense: N tensors, M GiB
dsv41 affine load: 309 linears packed, 4.30 GiB
AffineForwardNotPorted
```

The dense set is the norms, mHC mixers, router gate/bias, `wo_a`, this rank’s embed rows
and head rows; the linears are the packed projections. Both are freed before exit; the log
proves the affine load path fits without OOM.

## 5. If a smoke line fails

Save `summary.md`, `smoke.log`, and the relevant `cpu`/`y[0]` lines. Do not change the
checkpoint. The likely causes are:

- a pack-table or word-layout mismatch for that width;
- an expert view offset;
- a missing/unregistered environment variable;
- a fixture byte-copy regression (run `test_make_one_layer.py`).

Stop and send the failing module plus its CPU/GPU values.

## 6. Next after the sweep passes

The remaining Stage 1 gate is the one-layer forward. The sweep proves every layer-0
affine projection and expert stack; the forward still needs the attention and MoE
composition wired into the engine call graph. Do not start that in the same GPU window
as the first sweep unless the sweep is clean.
