"""Extract the video-only clips (modality 02) from downloaded RAVDESS actor archives.

Archives are expected in ../_datasets/ravdess/zips/Video_Speech_Actor_XX.zip
(downloaded from https://zenodo.org/records/1188976, CC BY-NC-SA 4.0).
The audio+video copies (modality 01) are skipped: they contain the same frames.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from common import RAVDESS, actor_dir


def extract(zip_path: Path) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not (name.startswith("02-") and name.endswith(".mp4")):
                continue
            actor = int(name[-6:-4])
            dst = actor_dir("videos", actor) / name
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dst, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zips", type=Path, default=RAVDESS / "zips")
    args = ap.parse_args()
    for z in sorted(args.zips.glob("Video_*_Actor_*.zip")):
        print(f"{z.name}: {extract(z)} new clips")


if __name__ == "__main__":
    main()
