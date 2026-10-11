"""Import CuPy with the CUDA runtime from pip (nvidia-*-cu12 wheels in the venv).

CuPy 13 on Windows only looks at CUDA_PATH; the toolkit is not installed system-wide, so
the wheels' bin folders are added to the DLL search path first.
    from gpu import cp      # None if no GPU / CuPy
"""
from __future__ import annotations

import os
import site
from pathlib import Path

cp = None
for base in site.getsitepackages():
    nv = Path(base) / "nvidia"
    if nv.is_dir():
        for b in nv.glob("*/bin"):
            os.add_dll_directory(str(b))
            os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
try:
    import cupy as cp  # noqa: F811
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no GPU, driver or CuPy: callers fall back to numpy
    cp = None
