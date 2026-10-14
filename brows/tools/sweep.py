"""Run evaluate.py for many (config, actor) jobs in parallel.

Jobs file: one config per line, "<name> <evaluate.py args...>"; every config runs for
every actor given on the command line, each as its own process.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import OUT, REPO

PY = sys.executable


def run(job) -> str:
    name, actor, args = job
    log = OUT / "logs" / f"{name}_a{actor}.log"
    if (OUT / "eval" / name / f"actor_{actor:02d}.npz").exists():
        return f"skip {name} {actor}"
    cmd = [PY, str(REPO / "tools" / "evaluate.py"), "--config", name, "--actors", str(actor), *args]
    # One thread per job: with 24+ jobs, OpenCV/BLAS thread pools thrash the CPU.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENCV_FOR_THREADS_NUM="1", BROW_SINGLE_THREAD="1")
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return f"{'ok' if r.returncode == 0 else 'FAIL'} {name} {actor}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--actors", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("-j", type=int, default=24)
    args = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    configs = []
    for line in args.jobs.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, *rest = shlex.split(line)
            configs.append((name, rest))
    jobs = [(n, a, r) for a in args.actors for n, r in configs]
    with ThreadPoolExecutor(args.j) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
