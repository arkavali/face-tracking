# face-tracking

>***Work In Progress***


Face tracking for **Arka'vali**, a mechatronic Avali fursuit. Small IR cameras inside the
mask watch the wearer's eyes, brows and mouth; each camera sits on its own ESP32 node that
runs plain image maths on the chip and sends a few numbers to the suit hub, which drives the
ears, feathers, eyelids, beak and LEDs.

Why not an off-the-shelf face tracker: inside the mask the AR periscope optics hide most of
the face, and the cameras see it from very close and at steep angles, where full-face models
such as MediaPipe lose it. So each part of the face gets its own camera, view and method.

## Trackers

| Folder | Tracks | Camera / board | Method | Status |
|---|---|---|---|---|
| [eyes/](eyes/) | gaze, blink, squint | ESP32-S3 CAM, OV2640 66 deg no-IR, one per eye | pupil ellipse fit (starburst + RANSAC), continuously re-estimated eyeball model | firmware and simulator done, waiting for the S3 boards |
| [brows/](brows/) | brow raise / frown, left and right | AI-Thinker ESP32-CAM above the mirror holder | plain image maths on the brow patches, no neural net | runs live on the ESP32-CAM, 25 fps camera |
| [mouth/](mouth/) | jaw, lips | ESP32(-S3) CAM inside the beak, IR LEDs | small CNN (22 outputs) trained on prompted calibration recordings | net runs on a classic ESP32 at ~10 fps; LED lighting study done |

Each folder is self-contained: `firmware/` (PlatformIO), `tools/` (Python, PC side),
`docs/` (studies and results) and `.bat` launchers for Windows. Start with the folder's
README.

## Layout

```
face-tracking/
  eyes/  brows/  mouth/   one tracker each: firmware/, tools/, docs/, *.bat
  requirements.txt        one Python venv for all trackers (.venv at the repo root)
  THIRD_PARTY.md          bundled and referenced third-party code, models and data
```

Planned: `common/` for code shared by the nodes, first of all the suit CAN bus protocol
library, so every tracker reports to the hub the same way.

## Setup

- Python 3.9 venv at the repo root: `py -3.9 -m venv .venv`, then
  `.venv\Scripts\pip install -r requirements.txt`. All `.bat` launchers use it.
- [PlatformIO](https://platformio.org/) for the firmware (`flash_firmware.bat` in each folder).
- Datasets are not in the repo. Tools look in `../_datasets` next to the repo folder, or in
  the folder named by the `FACE_DATASETS` environment variable.
- Optional: an NVIDIA GPU with CuPy speeds up the simulators (`tools/gpu.py`); everything
  falls back to NumPy.

## Licence

Copyright (C) 2026 arkavali

Everything in this repository (code, firmware, docs and images) is free software: you can
redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version. See [LICENSE](LICENSE).

In short: commercial use is allowed, but anyone who distributes this work or a modified
version (including firmware inside a device) must keep the copyright notices and publish
the full source under the same licence.

Third-party models, datasets and packages are not part of this repository and keep their
own licences: see [THIRD_PARTY.md](THIRD_PARTY.md).
