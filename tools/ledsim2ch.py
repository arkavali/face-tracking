"""Alternating side LEDs as two input channels vs both LEDs on at once (simulation).

  python ledsim2ch.py        -> out/led/report_2ch.txt, out/led/2ch.png
Mouth shapes from lipshapes.py, 15 random mask shifts (+-5 mm per axis) per shape. Variants:
  lens   - one LED next to the lens, 1 channel
  both   - left + right LED on together, 1 channel
  alt    - left frame and right frame as 2 channels (one common, locked exposure)
  alt-n  - alt with each channel exposed on its own (auto exposure per frame)
  ratio  - both-on image + (L - R) / (L + R) as the second channel (shape cue that cancels albedo)
Run twice: plain skin, and with a random smooth albedo pattern per sample (skin, stubble, light
differences between days; same pattern in the L and R frames) plus +-15 % gain per LED.
Scores: mean / min log separation over the shape pairs of ledsim.PAIRS, and nearest-centroid
accuracy (leave-one-out over the shifts) for all shapes and for the lip shapes only.
"""
from __future__ import annotations

import cv2
import numpy as np

import ledsim
import lipshapes
from scene import Scene, to_np

N_SHIFT = 15
LIP_SHAPES = ["neutral", "pucker", "funnel", "pucker_up", "jaw"]
# side pairs from the layout sweep (x, mm above the slit, mm in front), and the lens LED
SIDES = {"mid": ((45, 11, 64), (-45, 11, 64)), "low": ((45, -9, 64), (-45, -9, 64)),
         "near": ((45, 11, 44), (-45, 11, 44))}


def to_world(p):
    return np.array([p[0], lipshapes.SLIT[1] + p[1], lipshapes.SLIT[2] + p[2]], float)


def small(img):
    return img.reshape(60, 2, 80, 2).mean((1, 3))


def features(imgs_by_led, variant, leds, roi):
    """imgs_by_led: {led: (n_shift, H, W)} for one shape -> (n_shift, F) features."""
    out = []
    for s in range(N_SHIFT):
        if variant in ("lens", "both"):
            img = sum(imgs_by_led[k][s] for k in leds)
            out.append(small(ledsim.auto_expose(img, roi)).ravel())
        elif variant == "alt":
            a, b = (imgs_by_led[k][s] for k in leds)
            sc = np.percentile(np.maximum(a, b)[roi], 98)        # one locked exposure for both
            out.append(np.concatenate([small(np.clip(a / sc, 0, 1)).ravel(), small(np.clip(b / sc, 0, 1)).ravel()]))
        elif variant == "ratio":
            a, b = (imgs_by_led[k][s] for k in leds)
            both = small(ledsim.auto_expose(a + b, roi)).ravel()
            sa, sb = small(a), small(b)
            r = (sa - sb) / (sa + sb + 1e-3 * (sa + sb).max())
            out.append(np.concatenate([both, 0.5 + 0.5 * r.ravel()]))
        else:   # alt-n
            a, b = (imgs_by_led[k][s] for k in leds)
            out.append(np.concatenate([small(ledsim.auto_expose(a, roi)).ravel(), small(ledsim.auto_expose(b, roi)).ravel()]))
    return np.array(out)


def scores(feat):
    names = list(feat)
    m = {k: v.mean(0) for k, v in feat.items()}
    var = {k: ((v - m[k]) ** 2).sum(1).mean() for k, v in feat.items()}
    seps = [np.log(((m[a] - m[b]) ** 2).sum() / (0.5 * (var[a] + var[b]) + 1e-9)) for a, b in ledsim.PAIRS]

    def acc(subset):
        ok = n = 0
        for k in subset:
            for i in range(N_SHIFT):
                cents = {}
                for j in subset:
                    v = feat[j]
                    cents[j] = (v.sum(0) - v[i]) / (len(v) - 1) if j == k else v.mean(0)
                pred = min(cents, key=lambda j: ((feat[k][i] - cents[j]) ** 2).sum())
                ok += pred == k
                n += 1
        return ok / n
    return float(np.mean(seps)), float(np.min(seps)), acc(names), acc(LIP_SHAPES), seps


def main():
    rng = np.random.default_rng(1)
    shifts = [np.zeros(3)] + [rng.uniform(-5, 5, 3) for _ in range(N_SHIFT - 1)]
    shapes, faces, lipw = lipshapes.make_shapes()
    mask_tris = Scene(("mask",)).tris
    leds = {"lens": ledsim.CAM + [0, -6, 0]}
    for name, (l, r) in SIDES.items():
        leds[name + "_L"], leds[name + "_R"] = to_world(l), to_world(r)
    imgs = {}            # shape -> led -> (n_shift, H, W)
    roi = None
    for sname, verts in shapes.items():
        ss = ledsim.ShapeScene(verts, faces, lipw, mask_tris, base=shapes["neutral"])
        imgs[sname] = {k: np.zeros((N_SHIFT, ledsim.H, ledsim.W), np.float32) for k in leds}
        for si, sh in enumerate(shifts):
            d = ledsim.camera_rays(ledsim.CAM + sh, ledsim.AIM + sh)
            g = ss.primary(ledsim.CAM + sh, d)
            if roi is None:
                roi = to_np(g["hit"]).reshape(ledsim.H, ledsim.W) & (to_np(g["p"][:, 1]).reshape(ledsim.H, ledsim.W) < 125)
            for k, p in leds.items():
                imgs[sname][k][si] = to_np(ss.shade(g, p + sh, ledsim.AIM + sh)).reshape(ledsim.H, ledsim.W)
        print("rendered", sname, flush=True)

    variants = [("lens", "lens", ["lens"])]
    for name in SIDES:
        variants += [(f"both {name}", "both", [name + "_L", name + "_R"]),
                     (f"alt {name}", "alt", [name + "_L", name + "_R"]),
                     (f"alt-n {name}", "alt-n", [name + "_L", name + "_R"]),
                     (f"ratio {name}", "ratio", [name + "_L", name + "_R"])]
    # albedo pattern + per-LED gain per sample (same pattern under both LEDs of a sample)
    tex = {}
    for sname in shapes:
        fields = []
        for si in range(N_SHIFT):
            f = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), (ledsim.W, ledsim.H), interpolation=cv2.INTER_CUBIC)
            fields.append(np.clip(1 + 0.25 * f, 0.4, 1.8))
        tex[sname] = np.array(fields)
    gains = {sname: {k: rng.uniform(0.85, 1.15, N_SHIFT) for k in leds} for sname in shapes}
    textured = {s: {k: v * tex[s] * gains[s][k][:, None, None] for k, v in imgs[s].items()} for s in shapes}
    lines = []
    for title, data in (("plain skin", imgs), ("albedo pattern + LED gain noise", textured)):
        lines += [f"== {title}", "variant          sep mean   sep min   acc all   acc lips   per pair: "
                  + ", ".join(f"{a}/{b}" for a, b in ledsim.PAIRS)]
        for label, kind, ks in variants:
            feat = {s: features(data[s], kind, ks, roi) for s in shapes}
            mean, mn, aa, al, seps = scores(feat)
            lines.append(f"{label:15s} {mean:8.2f} {mn:9.2f} {aa:9.2f} {al:10.2f}   " + " ".join(f"{v:5.1f}" for v in seps))
        lines.append("")
    pos = "; ".join(f"{n}: L {SIDES[n][0]} R {SIDES[n][1]}" for n in SIDES)
    lines += ["", f"{N_SHIFT} mask shifts (+-5 mm) per shape; LED positions (x, mm above slit, mm in front): {pos}",
              "sep = log(between-shape distance / within-shape spread from the shifts) on the 80x60 net input",
              "acc = nearest-centroid accuracy, leave-one-out over the shifts (lips: neutral/pucker/funnel/pucker_up/jaw)"]
    (ledsim.OUT / "report_2ch.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # picture: for each lip shape, left frame | right frame | both on (shift 0)
    rows = []
    for s in LIP_SHAPES + ["smile", "cheeks"]:
        a, b = imgs[s]["mid_L"][0], imgs[s]["mid_R"][0]
        sc = np.percentile(np.maximum(a, b)[roi], 98)
        tiles = [np.clip(a / sc, 0, 1), np.clip(b / sc, 0, 1), ledsim.auto_expose(a + b, roi)]
        t = [cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for x in tiles]
        cv2.putText(t[0], s, (3, 12), 0, 0.4, (0, 255, 255), 1)
        rows.append(np.hstack(t))
    cv2.imwrite(str(ledsim.OUT / "2ch.png"), cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
