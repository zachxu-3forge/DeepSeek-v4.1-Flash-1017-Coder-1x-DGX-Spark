#!/usr/bin/env python3
"""Compare two captured head-logit sets with the MoE correctness rule.

Rule (Implementation.md D27): over a fixed prompt set and every generated
decode step, top-1 agreement must be at least 99% and mean KL(reference ||
candidate) at most 0.005.  The reference is the frozen host-MoE build; the
candidate is any later optimization.

Input directories come from `capture_logits.py`.  Each step file is a raw
little-endian float32 logits row (the server traces id 65521 before the
sampler).
"""

import argparse
import array
import json
import math
import pathlib
import sys


def logits(path: pathlib.Path) -> array.array:
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // 4)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def score(ref: array.array, cand: array.array) -> tuple[int, float]:
    if len(ref) != len(cand) or not ref:
        raise ValueError(f"logit rows differ: {len(ref)} vs {len(cand)}")
    best_ref = max(range(len(ref)), key=ref.__getitem__)
    best_cand = max(range(len(cand)), key=cand.__getitem__)
    mref = ref[best_ref]
    mcand = cand[best_cand]
    sref = 0.0
    scand = 0.0
    for value in ref:
        sref += math.exp(value - mref)
    for value in cand:
        scand += math.exp(value - mcand)
    lzref = mref + math.log(sref)
    lzcand = mcand + math.log(scand)
    kl = 0.0
    for a, b in zip(ref, cand):
        la = a - lzref
        kl += math.exp(la) * (la - (b - lzcand))
    return int(best_ref == best_cand), max(kl, 0.0)


def step_files(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(directory.glob("logits_*.bin"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=pathlib.Path, required=True)
    parser.add_argument("--cand", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--min-top1", type=float, default=0.99)
    parser.add_argument("--max-kl", type=float, default=0.005)
    args = parser.parse_args()

    prompt_dirs = sorted(p for p in args.ref.iterdir() if p.is_dir())
    if not prompt_dirs:
        raise SystemExit(f"no prompt directories under {args.ref}")

    total_steps = 0
    total_top1 = 0
    total_kl = 0.0
    max_kl = 0.0
    prompts: list[dict] = []

    for ref_dir in prompt_dirs:
        cand_dir = args.cand / ref_dir.name
        if not cand_dir.is_dir():
            raise SystemExit(f"missing candidate prompt directory {cand_dir}")
        ref_files = step_files(ref_dir)
        cand_files = step_files(cand_dir)
        if len(ref_files) != len(cand_files):
            raise SystemExit(
                f"{ref_dir.name}: {len(ref_files)} reference steps, {len(cand_files)} candidate steps"
            )
        if not ref_files:
            raise SystemExit(f"{ref_dir.name}: no logits captured")
        hits = 0
        kl_sum = 0.0
        prompt_max = 0.0
        for ref_path, cand_path in zip(ref_files, cand_files):
            hit, kl = score(logits(ref_path), logits(cand_path))
            hits += hit
            kl_sum += kl
            prompt_max = max(prompt_max, kl)
        steps = len(ref_files)
        total_steps += steps
        total_top1 += hits
        total_kl += kl_sum
        max_kl = max(max_kl, prompt_max)
        prompts.append(
            {
                "id": ref_dir.name,
                "steps": steps,
                "top1_agreement": hits / steps,
                "mean_kl": kl_sum / steps,
                "max_kl": prompt_max,
            }
        )
        print(
            f"{ref_dir.name:16s} steps={steps:4d} top1={hits / steps:7.4%} "
            f"mean_kl={kl_sum / steps:.6f} max_kl={prompt_max:.6f}"
        )

    top1 = total_top1 / total_steps
    mean_kl = total_kl / total_steps
    passed = top1 >= args.min_top1 and mean_kl <= args.max_kl
    result = {
        "reference": str(args.ref),
        "candidate": str(args.cand),
        "rule": {"min_top1": args.min_top1, "max_kl": args.max_kl},
        "steps": total_steps,
        "top1_agreement": top1,
        "mean_kl": mean_kl,
        "max_kl": max_kl,
        "pass": passed,
        "prompts": prompts,
    }
    print(
        f"TOTAL steps={total_steps} top1={top1:.4%} mean_kl={mean_kl:.6f} "
        f"max_kl={max_kl:.6f} pass={passed}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
