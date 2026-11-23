# Storage layout

Checkpoints and run artifacts live inside the checkout, not in the home directory:

```text
$ROOT/
  checkpoints/
    base/                 # original MLX-Coder safetensors and config (255 GB apparent)
    model-translated/     # served affine overlay (21 shards, hardlinked to base)
    engram-native/        # native 4-bit Engram tables (115 GB)
    assets/               # tokenizer, AOT cubins, token map
  work/
    correctness/          # logits traces and comparisons
    legacy-dsv41-mlx/     # earlier profiling and stage-1 scratch
    logs/                 # archived host logs
```

`checkpoints/` and `work/` are in `.gitignore`, and both are excluded by
`.dockerignore`, so a build never sends the weights as context.

`base` and `model-translated` share every weight shard as a hardlink (same inode,
link count 2). Deleting one does not free those blocks until both names are gone.
A `du` that lists one directory alone can therefore look smaller than the pair;
`stat -c '%h %i %n'` shows the shared inode.

The one-Spark affine server mounts only `checkpoints/model-translated`,
`checkpoints/engram-native`, and `checkpoints/assets`:

```sh
ROOT=$(git rev-parse --show-toplevel)
docker run -d --rm --name dsv41-serve --platform linux/arm64 \
  --gpus all --ipc=host --shm-size=16g --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 -p 127.0.0.1:8888:8080 \
  -v "$ROOT/checkpoints/model-translated:/model:ro" \
  -v "$ROOT/checkpoints/engram-native:/engram-native:ro" \
  -v "$ROOT/checkpoints/assets:/assets:ro" \
  -e TF_DSV41_ASSETS=/assets -e TF_DSV41_ENGRAM_DIR=/engram-native \
  -e TF_DSV41_AFFINE_LOAD=1 -e TF_DSV41_AFFINE_MAX_GIB=110 \
  -e TF_DSV41_GRAPHS=1 -e TF_DSV41_CONTEXT=32768 \
  -e TF_DSV41_SAMPLING=1 -e TF_DSV41_ENGRAM_WARM=1 \
  -e TF_EXPERIMENTAL=deepseek -e TF_TP_WORLD=1 tf-dsv41-mlx:dev \
  tensorfold-dsv41 serve /model --no-drafts --host 0.0.0.0
```

The older path names in historical notes (`/home/deploy/dsv41-mlx`,
`/home/deploy/Models/DeepSeek-v4.1-Flash-1017/MLX-Coder`) refer to this layout
before it moved; the commands are otherwise unchanged.
