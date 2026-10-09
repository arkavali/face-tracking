"""Build out/report.html: metrics table for every evaluated config plus plots and frames."""
from __future__ import annotations

import html
import json

from common import OUT

COLS = [
    ("r_gt", "r vs MP (camera view)", "{:.2f}"),
    ("rmse_mm", "RMSE, mm", "{:.2f}"),
    ("r_mp", "r vs MP (head frame)", "{:.2f}"),
    ("r_inner", "r inner", "{:.2f}"),
    ("r_outer", "r outer", "{:.2f}"),
    ("noise_mm", "rest jitter, mm", "{:.2f}"),
    ("span_mm", "MP span, mm", "{:.1f}"),
    ("snr", "span / jitter", "{:.1f}"),
    ("valid", "valid", "{:.0%}"),
    ("vis", "contour visible", "{:.0%}"),
]


def describe(cfg: dict) -> str:
    r, lk, t = cfg["rig"], cfg["look"], cfg["tracker"]
    cam = (f"cam y{r['cam_y']:.0f}/z{r['cam_z']:.0f} mm" if "cam_y" in r
           else f"cam +{r['cam_up']:.0f}/{r['cam_fwd']:+.0f} vs flange tip (old slab holder)")
    parts = [f"{r['mode']}", cam,
             f"holder top {r['flange_top']:.0f} mm", f"{r['width']}x{r['height']} {r['dfov_deg']:.0f}deg",
             f"lights {'+'.join(r['lights'])}", f"IR contrast {lk['ir_contrast']}", f"noise {lk['noise_read']}",
             f"method {t['method']}"]
    if cfg.get("glasses"):
        parts.append("glasses")
    return ", ".join(parts)


def main() -> None:
    rows, sections = [], []
    for summ in sorted((OUT / "eval").glob("*/summary.json")):
        d = json.loads(summ.read_text())
        cdir = summ.parent
        cfg_path = cdir / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
        name = d["config"]
        desc = describe(cfg) if cfg else ""
        n_act = len(d["per_actor"])
        cells = "".join(f"<td>{fmt.format(d['median'][k])}</td>" for k, _, fmt in COLS)
        rows.append(f"<tr><td><a href='#{name}'>{html.escape(name)}</a></td><td>{n_act}</td>{cells}"
                    f"<td class='desc'>{html.escape(desc)}</td></tr>")
        imgs = "".join(f"<img src='eval/{name}/{p.name}' loading='lazy'>" for p in sorted(cdir.glob("plot_*.png")))
        imgs += "".join(f"<img src='eval/{name}/{p.name}' loading='lazy'>" for p in sorted(cdir.glob("sheet*.png")))
        per = "".join(
            f"<tr><td>{a}</td><td>{s}</td>" + "".join(f"<td>{fmt.format(v[s][k])}</td>" for k, _, fmt in COLS) + "</tr>"
            for a, v in d["per_actor"].items() for s in "RL")
        sections.append(f"<section id='{name}'><h2>{html.escape(name)}</h2><p>{html.escape(desc)}</p>"
                        f"<table class='small'><tr><th>actor</th><th>brow</th>"
                        + "".join(f"<th>{h}</th>" for _, h, _ in COLS) + f"</tr>{per}</table>{imgs}</section>")
    head = "".join(f"<th>{h}</th>" for _, h, _ in COLS)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Brow Tracker Sims</title>
<style>
:root {{ --bg:#fafafa; --fg:#1d1d1f; --mut:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --fg:#e8e8ea; --mut:#9a9aa0; --line:#333; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:16px; }}
table {{ border-collapse:collapse; margin:8px 0 16px; }}
td, th {{ border:1px solid var(--line); padding:3px 7px; text-align:right; }}
td:first-child, th:first-child {{ text-align:left; }}
td.desc {{ text-align:left; color:var(--mut); font-size:12px; }}
table.small td, table.small th {{ font-size:12px; }}
img {{ max-width:100%; display:block; margin:8px 0; border:1px solid var(--line); }}
h2 {{ margin-top:32px; }}
</style></head><body>
<h1>Brow tracker - simulated mask camera vs MediaPipe</h1>
<p>Medians over actors and both brows. "MP" = MediaPipe FaceLandmarker on the original frontal video (the teacher).
Positive raise = brow up. See tools/metrics.py for definitions.</p>
<table><tr><th>config</th><th>actors</th>{head}<th>setup</th></tr>{''.join(rows)}</table>
{''.join(sections)}
</body></html>"""
    (OUT / "report.html").write_text(page, encoding="utf-8")
    print(OUT / "report.html")


if __name__ == "__main__":
    main()
