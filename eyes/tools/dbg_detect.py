"""Debug sheet: detector edge points and ellipse on the preview states."""
import sys
import cv2
import numpy as np
import eyesim as E
import pupil as P
from placements import make, NAMES
from preview import STATES

det = P.PupilDetector()
rows = []
for name in (sys.argv[2:] or NAMES):
    sc, cam, _ = make(name)
    r = E.Renderer(sc, cam)
    tiles = []
    for lab, st in STATES:
        img, gt = r.render(st)
        t = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        seed, lev = P.darkest_seed(img, det.p.box)
        pts = P.starburst(img, seed, lev, det.p)
        for q in pts:
            cv2.circle(t, (int(q[0]), int(q[1])), 1, (0, 255, 0), -1)
        e = det.detect(img)
        if e is not None:
            cv2.ellipse(t, ((e.cx, e.cy), (2 * e.a, 2 * e.b), np.degrees(e.angle)), (0, 0, 255), 1)
        if "pc_full" in gt:
            cv2.drawMarker(t, tuple(int(v) for v in gt["pc_full"]), (255, 0, 255), cv2.MARKER_CROSS, 6)
        cv2.putText(t, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t[40:200, 60:260])
    rows.append(np.hstack(tiles))
cv2.imwrite(sys.argv[1], cv2.resize(np.vstack(rows), None, fx=1.5, fy=1.5))
