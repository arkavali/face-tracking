"""Download the MediaPipe Face Landmarker model (Apache-2.0, Google) to models/."""
from __future__ import annotations

import urllib.request

from common import MODEL

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/latest/face_landmarker.task")

if __name__ == "__main__":
    if MODEL.exists():
        print("already there:", MODEL)
    else:
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, MODEL)
        print("saved", MODEL, MODEL.stat().st_size, "bytes")
