# patches-mlx

Engine patches applied in filename order to upstream TensorFold (`TF_REF` in
`docker/Dockerfile.mlx`). The current set:

| Patch | Purpose |
| --- | --- |
| `0001-admit-omlx-affine.patch` | Read `omlx_deepseek_v41.quantized_modules`, translate it to the engine's affine block, and admit `quant_method = "affine"` in the dsv41 config path |
| `0002-dsv41-affine-layout.patch` | Resolve translated affine metadata and validate packed U32/scales/biases shapes (host-only) |
| `0003-dsv41-affine-tensor-reader.patch` | Read U32/BF16 named images, preserve stacked experts, and keep the whole `wo_a` |
| `0004-dsv41-affine-plan.patch` | Enumerate affine linears and expert stacks from headers; default unlisted modules to dense |
| `0005-dsv41-affine-boot-guard.patch` | Build and validate the affine plan at boot; return `AffineForwardNotPorted` before the EXL3 path |
| `0006-dsv41-192-world1.patch` | Exercise a 192-expert affine plan and single-rank admission |
| `0007-dsv41-mixed-dspark.patch` | List DSpark MXFP8/MXFP4 modules as mixed and log them at boot |
| `0008-dsv41-affine-reference.patch` | CPU reference for the MLX bit-stream dequantization and row dot |
| `0009-dsv41-affine-cuda-audit.patch` | Report native CUDA width support; 2-bit routed experts need a per-expert fallback |
| `0010-dsv41-expert-fallback.patch` | Per-expert byte views into stacked affine projections for the `AffineN` fallback |
| `0011-dsv41-world1.patch` | Admit and run the single-rank serving path end to end |
| `0012-dsv41-engram-affine.patch` | Read or convert the target's 4-bit/group-32 Engram tables |

Implemented: `0001`-`0010`. Planned: `0011`-`0012`.

Rules:

- One logical change per patch; apply cleanly with `git apply --check`.
- Rebase onto the pinned `TF_REF` before adding a new patch; do not stack unrelated changes.
- Keep the upstream file layout; put new family code under `zig/src/families/deepseek_v41/`.
- Record the upstream commit and the patch set in `Agents/Implementation.md`.
