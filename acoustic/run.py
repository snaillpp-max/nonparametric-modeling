"""Run the four-stage acoustic examples using only files in this repository."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

EXAMPLES = ("pulsating_sphere", "piston_duct", "westervelt_cavity")
ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("all",) + EXAMPLES, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--stage", choices=("all", "generate", "pod", "train", "predict"), default="all")
    parser.add_argument("--output", type=Path, required=True, help="Writable output directory; separate subdirectories are created for each example.")
    parser.add_argument("--threads", type=int, default=2, help="Maximum BLAS/Numba worker threads per example.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = args.output.expanduser().resolve()
    # Do not mix generated numerical files into the source checkout.
    source = ROOT / "examples"
    if output == source or source in output.parents:
        parser.error("Choose an output directory outside acoustic/examples")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    selected = EXAMPLES if args.example == "all" else (args.example,)
    records = []
    for example in selected:
        destination = output / example
        destination.mkdir(parents=True, exist_ok=True)
        cache = destination / ".cache"
        for name in ("tmp", "matplotlib", "numba"):
            (cache / name).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({key: str(args.threads) for key in
                    ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")})
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLBACKEND="Agg",
                   ACOUSTIC_MAX_THREADS=str(args.threads),
                   TMP=str(cache / "tmp"), TEMP=str(cache / "tmp"), TMPDIR=str(cache / "tmp"),
                   MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"))
        script = ROOT / "examples" / example / "run.py"
        command = [sys.executable, "-B", str(script), "--profile", args.profile,
                   "--stage", args.stage, "--output", str(destination)]
        print(f"\n{example}: {args.profile}, {args.stage}", flush=True)
        start = time.perf_counter()
        log = destination / f"invocation_{stamp}.log"
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", env=env)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = proc.wait()
        records.append({"example": example, "profile": args.profile, "stage": args.stage,
                        "returncode": code, "elapsed_seconds": time.perf_counter() - start,
                        "entrypoint_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                        "log": log.name})
        report = {"schema": 1, "utc": stamp, "python": platform.python_version(),
                  "profile": args.profile, "threads_per_process": args.threads, "runs": records,
                  "scope": "Four-stage execution check; smoke results do not certify manuscript accuracy."}
        (output / f"invocation_{stamp}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if code:
            print(f"Stopped after {example}; see {log}", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
