# patches-mlx

Engine patches applied in filename order to upstream TensorFold (`TF_REF` in
`docker/Dockerfile.mlx`). Phase 0 starts with an empty set; the first patches will be named:

| Patch | Purpose |
| --- | --- |
| `0001-admit-omlx-affine.patch` | Read `omlx_deepseek_v41.quantized_modules`, translate it to the engine's affine block, and admit `quant_method = "affine"` in the dsv41 config path |
| `0002-dsv41-affine-layout.patch` | Resolve translated affine metadata and validate packed U32/scales/biases shapes (host-only) |
| `0003-dsv41-affine-tensor-reader.patch` | Read U32/BF16 named images, preserve stacked experts, and keep the whole `wo_a` |
| `0004-dsv41-192-experts.patch` | Make expert count, top-k, layouts, and DSpark expert counts data-driven |
| `0005-dsv41-world1.patch` | Admit and run the single-rank serving path |
| `0006-dsv41-engram-affine.patch` | Read or convert the target's 4-bit/group-32 Engram tables |

Implemented: `0001-admit-omlx-affine.patch`, `0002-dsv41-affine-layout.patch`, and `0003-dsv41-affine-tensor-reader.patch`. `0004`-`0006` are planned.

Rules:

- One logical change per patch; apply cleanly with `git apply --check`.
- Rebase onto the pinned `TF_REF` before adding a new patch; do not stack unrelated changes.
- Keep the upstream file layout; put new family code under `zig/src/families/deepseek_v41/`.
- Record the upstream commit and the patch set in `Agents/Implementation.md`.
