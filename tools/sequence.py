"""Synthetic gaze / lid / slip sequences at 25 fps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eyesim import EyeState

FPS = 25
CALIB_GRID = [(y, p) for p in (15, 0, -15) for y in (-20, 0, 20)]


@dataclass
class Frame:
    st: EyeState
    phase: str              # "calib", "free"
    target: int             # calibration target index or -1
    settled: bool           # calibration: gaze has settled on the target
    slip: tuple             # (dpos (3,), drot (3,)) camera slip relative to the head


def make_sequence(seconds: float = 60.0, seed: int = 0, slip_mm=(1.0, -1.5, 1.0),
                  slip_deg=(2.0, 0.0, 1.0), slip_at: float = 0.5) -> list:
    rng = np.random.default_rng(seed)
    frames = []
    pr = 2.5
    zero = (np.zeros(3), np.zeros(3))
    # calibration: 9 targets, 1.2 s each, the first 0.4 s are the saccade + settling
    for i, (y, p) in enumerate(CALIB_GRID):
        for k in range(int(1.2 * FPS)):
            frames.append(Frame(EyeState(y, p, pr), "calib", i, k >= int(0.4 * FPS), zero))
    # free viewing
    n = int(seconds * FPS)
    yaw, pitch = 0.0, 0.0
    t = 0
    states = []
    while len(states) < n:
        if rng.random() < 0.2:     # smooth pursuit
            dur = int(rng.uniform(1.5, 3.0) * FPS)
            amp_y, amp_p = rng.uniform(10, 25), rng.uniform(0, 15)
            fr = rng.uniform(0.2, 0.5)
            ph = rng.uniform(0, 6.28)
            y0, p0 = yaw - amp_y * np.sin(ph), pitch - amp_p * np.sin(ph * 0.7)
            for k in range(dur):
                tt = k / FPS
                yaw = float(np.clip(y0 + amp_y * np.sin(2 * np.pi * fr * tt + ph), -32, 32))
                pitch = float(np.clip(p0 + amp_p * np.sin(2 * np.pi * fr * 0.7 * tt + ph * 0.7), -22, 22))
                states.append([yaw, pitch])
        else:                      # saccade + fixation
            ny, npi = rng.uniform(-30, 30), rng.uniform(-20, 20)
            states.append([(yaw + ny) / 2, (pitch + npi) / 2])
            yaw, pitch = ny, npi
            for _ in range(int(rng.uniform(0.3, 1.5) * FPS)):
                states.append([yaw + rng.normal(0, 0.15), pitch + rng.normal(0, 0.15)])
        t += 1
    states = np.array(states[:n])
    openness = np.ones(n)
    squint = np.zeros(n)
    k = int(rng.uniform(1, 3) * FPS)
    while k < n - 6:
        prof = [0.6, 0.1, 0.0, 0.25, 0.7]
        openness[k:k + 5] = np.minimum(openness[k:k + 5], prof[: len(openness[k:k + 5])])
        k += int(rng.uniform(2.5, 5.5) * FPS)
    for _ in range(2):
        a = int(rng.uniform(0.1, 0.9) * n)
        d = int(2.0 * FPS)
        ramp = np.clip(np.minimum(np.arange(d), d - 1 - np.arange(d)) / 5, 0, 1)
        sl = slice(a, min(a + d, n))
        squint[sl] = np.maximum(squint[sl], 0.7 * ramp[: sl.stop - sl.start])
        openness[sl] = np.minimum(openness[sl], 1 - 0.45 * ramp[: sl.stop - sl.start])
    pr_t = np.clip(2.5 + np.cumsum(rng.normal(0, 0.03, n)), 1.5, 3.5)
    s0 = int(slip_at * n)
    ramp = np.clip((np.arange(n) - s0) / (2 * FPS), 0, 1)
    for i in range(n):
        slip = (np.asarray(slip_mm) * ramp[i], np.asarray(slip_deg) * ramp[i])
        st = EyeState(states[i, 0], states[i, 1], float(pr_t[i]), float(openness[i]), float(squint[i]))
        frames.append(Frame(st, "free", -1, True, slip))
    return frames
