#!/usr/bin/env python3
"""Capture every decode step's head logits from a running TensorFold server.

The server must be started with all three settings:

    TF_DSV41_LAYER_STATS=1
    TF_DSV41_TRACE_SEQ=1
    TF_DSV41_TRACE_DIR=/some/container/path

`TF_DSV41_TRACE_SEQ=1` makes the head-logits trace (id 65521) append a sequence
number, so a request leaves one file per generated token instead of overwriting
one file.  This script sends a fixed prompt file, copies each request's trace
files into `<out>/<id>/logits_NNNNN.bin`, and records the response metadata.
"""

import argparse
import json
import pathlib
import shutil
import time
import urllib.request


def trace_files(trace_dir: pathlib.Path, trace_id: int) -> dict[int, pathlib.Path]:
    found: dict[int, pathlib.Path] = {}
    prefix = f"trace_{trace_id}_"
    for path in trace_dir.glob(prefix + "*.bin"):
        suffix = path.name[len(prefix) : -len(".bin")]
        try:
            found[int(suffix)] = path
        except ValueError:
            continue
    return found


def post(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18941")
    parser.add_argument("--trace-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--prompts", type=pathlib.Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trace-id", type=int, default=0xFFF1)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    for item in prompts:
        prompt_id = item["id"]
        before = trace_files(args.trace_dir, args.trace_id)
        started = time.time()
        response = post(
            url,
            {
                "model": "model",
                "messages": [{"role": "user", "content": item["prompt"]}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            args.timeout,
        )
        elapsed = time.time() - started
        completion_tokens = int(response.get("usage", {}).get("completion_tokens", 0))
        after = trace_files(args.trace_dir, args.trace_id)
        new = sorted(
            ((seq, path) for seq, path in after.items() if seq not in before),
            key=lambda pair: pair[0],
        )
        if len(new) < completion_tokens:
            raise SystemExit(
                f"{prompt_id}: expected {completion_tokens} new trace files, found {len(new)}; "
                "check TF_DSV41_TRACE_SEQ / TF_DSV41_LAYER_STATS in the server"
            )

        # Prefill may trace a head row too; the last completion_tokens files are the decode steps.
        chosen = new[-completion_tokens:] if completion_tokens else []
        target = args.out / prompt_id
        target.mkdir(parents=True, exist_ok=True)
        for index, (_, path) in enumerate(chosen):
            shutil.copy2(path, target / f"logits_{index:05d}.bin")

        tensorfold = response.get("tensorfold", {})
        meta = {
            "id": prompt_id,
            "prompt": item["prompt"],
            "max_tokens": args.max_tokens,
            "elapsed_s": elapsed,
            "completion_tokens": completion_tokens,
            "token_sha": tensorfold.get("token_sha"),
            "tokens_per_second": tensorfold.get("tokens_per_second"),
            "time_to_first_token": tensorfold.get("time_to_first_token"),
            "content": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "trace_sequences": [seq for seq, _ in chosen],
            "trace_bytes": [path.stat().st_size for _, path in chosen],
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"{prompt_id:16s} tokens={completion_tokens:3d} sha={meta['token_sha']} "
            f"files={len(chosen)} elapsed={elapsed:.2f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
