"""Per-stage timing of the detector on the S3 bench board: python hw_profile.py COM7 [seq.npz]"""
import sys, numpy as np
sys.path.insert(0, ".")
from hw_bench import open_port, ask
from parity_test import frames_from
imgs, f = frames_from(sys.argv[2] if len(sys.argv) > 2 else str(__import__("pathlib").Path(__file__).resolve().parent.parent / "out" / "seq" / "thru_ledside.npz"), 60)
s = open_port(sys.argv[1])
print(ask(s, "reset", prefix="OK"))
for i in (5, 50):
    print(ask(s, "F 320 240", imgs[i].tobytes(), prefix="R ")[:60])
    print(ask(s, "bench 5", prefix="B"))
