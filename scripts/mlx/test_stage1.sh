#!/usr/bin/env bash
# Stage 1 verification harness. Runs on the Spark host (or any host with the model)
# and optionally drives the Docker image. It records everything under OUT_DIR so the
# deployment can be reviewed later.
#
# Usage:
#   scripts/mlx/test_stage1.sh --host-only
#   CONTAINER=1 scripts/mlx/test_stage1.sh
#   CONTAINER=1 EXPECT_FORWARD=1 scripts/mlx/test_stage1.sh
#
# Environment:
#   MODEL_DIR   checkpoint directory (required)
#   OUT_DIR     result directory (default out/stage1/<utc timestamp>)
#   IMAGE       image tag (default tf-dsv41-mlx:dev)
#   CONTAINER=1 also run the engine in the image
#   BUILD=1     build the image before the container run
#   EXPECT_FORWARD=1  require the one-layer forward to pass (after stage 1m)
#   SERVE_ARGS  engine arguments (default "serve /model")
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
MODEL_DIR=${MODEL_DIR:-}
CONTAINER=${CONTAINER:-0}
EXPECT_FORWARD=${EXPECT_FORWARD:-0}
SMOKE=${SMOKE:-0}
SERVE_ARGS=${SERVE_ARGS:-serve /model}
IMAGE=${IMAGE:-tf-dsv41-mlx:dev}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR=${OUT_DIR:-$ROOT/out/stage1/$STAMP}

for arg in "$@"; do
  case "$arg" in
    --host-only) CONTAINER=0 ;;
    --container) CONTAINER=1 ;;
    --expect-forward) EXPECT_FORWARD=1 ;;
    --smoke) SMOKE=1 ;;
    --build) BUILD=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
  echo "MODEL_DIR must point at the checkpoint directory" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
fail=0
declare -a lines=()

step() {
  local name=$1
  shift
  echo "== $name"
  "$@" >"$OUT_DIR/$name.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    lines+=("| $name | PASS | \`$name.log\` |")
  else
    lines+=("| $name | FAIL ($rc) | \`$name.log\` |")
    fail=1
  fi
  return $rc
}

echo "stage 1 verification: $STAMP"
echo "model: $MODEL_DIR"
echo "out:   $OUT_DIR"

step inspect_checkpoint "$PY" "$ROOT/scripts/mlx/inspect_checkpoint.py" "$MODEL_DIR" --json "$OUT_DIR/inspect.json"
step translate_config "$PY" "$ROOT/scripts/mlx/translate_config.py" "$MODEL_DIR" --check \
  --out "$OUT_DIR/quantization.json" --write-config "$OUT_DIR/config.translated.json"
step plan_dispatch "$PY" "$ROOT/scripts/mlx/plan_dispatch.py" "$MODEL_DIR" --json "$OUT_DIR/plan_dispatch.json"

if [[ -f "$MODEL_DIR/model-00001-of-00021.safetensors" && -f "$MODEL_DIR/model-00020-of-00021.safetensors" ]]; then
  step reference_layer "$PY" "$ROOT/scripts/mlx/reference_layer.py" "$MODEL_DIR" --out "$OUT_DIR/reference_layer.json"
else
  lines+=("| reference_layer | SKIP | shard 20 not downloaded |")
fi
if [[ -f "$MODEL_DIR/model-00018-of-00021.safetensors" && -f "$MODEL_DIR/model-00019-of-00021.safetensors" ]]; then
  step engram "$PY" "$ROOT/scripts/mlx/test_engram.py" "$MODEL_DIR" --out "$OUT_DIR/engram.json"
else
  lines+=("| engram | SKIP | Engram shards not downloaded |")
fi

if [[ "$CONTAINER" == "1" ]]; then
  if [[ "${BUILD:-0}" == "1" ]]; then
    step build_image env BUILD=1 MODEL_DIR="$MODEL_DIR" IMAGE="$IMAGE" "$ROOT/scripts/mlx/run_container.sh" tensorfold-dsv41 --version
  fi
  echo "== engine"
  set +e
  TF_DSV41_LAYERS=0-0 TF_DSV41_DRAFTS=0 \
    timeout 180 "$ROOT/scripts/mlx/run_container.sh" tensorfold-dsv41 $SERVE_ARGS \
    >"$OUT_DIR/engine.log" 2>&1
  rc=$?
  set -e
  if grep -q "AffineForwardNotPorted" "$OUT_DIR/engine.log"; then
    if [[ "$EXPECT_FORWARD" == "1" ]]; then
      lines+=("| engine | FAIL | affine forward still guarded |")
      fail=1
    else
      lines+=("| engine | PASS (guard as expected) | \`engine.log\` |")
    fi
  elif grep -q "dsv41 affine:" "$OUT_DIR/engine.log"; then
    if [[ "$EXPECT_FORWARD" == "1" ]]; then
      lines+=("| engine | PASS | \`engine.log\` |")
    else
      lines+=("| engine | PASS (no guard; forward may be active) | \`engine.log\` |")
    fi
  else
    lines+=("| engine | FAIL ($rc) | \`engine.log\` |")
    fail=1
  fi

  if [[ "$SMOKE" == "1" ]]; then
    echo "== affine smoke"
    set +e
    TF_DSV41_LAYERS=0-0 TF_DSV41_DRAFTS=0 TF_DSV41_AFFINE_SMOKE=1 \
      timeout 300 "$ROOT/scripts/mlx/run_container.sh" tensorfold-dsv41 $SERVE_ARGS \
      >"$OUT_DIR/smoke.log" 2>&1
    rc_smoke=$?
    set -e
    if grep -q "affine smoke" "$OUT_DIR/smoke.log"; then
      lines+=("| affine smoke | PASS | \`smoke.log\` |")
    else
      lines+=("| affine smoke | FAIL ($rc_smoke) | \`smoke.log\` |")
      fail=1
    fi
  fi

  if [[ -f "$OUT_DIR/plan_dispatch.json" && -f "$OUT_DIR/engine.log" ]]; then
    if "$PY" - "$OUT_DIR/plan_dispatch.json" "$OUT_DIR/engine.log" <<'PY'
import json, re, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))["counts"]
text = open(sys.argv[2], encoding="utf-8", errors="replace").read()
m = re.search(r"dispatch: (\d+) affine8 linears, (\d+) expert fallbacks, (\d+) mxfp8, (\d+) mxfp4", text)
if not m:
    print("no dispatch line in the engine log", file=sys.stderr)
    raise SystemExit(1)
got = {
    "affine8_linear": int(m.group(1)),
    "affine_n_expert": int(m.group(2)),
    "mxfp8": int(m.group(3)),
    "mxfp4": int(m.group(4)),
}
if got != {k: plan[k] for k in got}:
    print(f"dispatch mismatch: plan={plan} engine={got}", file=sys.stderr)
    raise SystemExit(1)
print("dispatch counts match")
PY
    then
      lines+=("| dispatch match | PASS | plan vs engine log |")
    else
      lines+=("| dispatch match | FAIL | \`engine.log\` |")
      fail=1
    fi
  fi
else
  lines+=("| engine | SKIP | CONTAINER not set |")
fi

{
  echo "# Stage 1 verification: $STAMP"
  echo
  echo "Model: \`$MODEL_DIR\`"
  echo
  echo "| Check | Result | Log |"
  echo "|---|---|---|"
  printf '%s\n' "${lines[@]}"
  echo
  if [[ $fail -eq 0 ]]; then
    echo "Overall: **PASS**"
  else
    echo "Overall: **FAIL**"
  fi
} >"$OUT_DIR/summary.md"

cat "$OUT_DIR/summary.md"
exit $fail
