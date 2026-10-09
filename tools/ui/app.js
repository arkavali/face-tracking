// Brow Camera UI: talks to browui.py (serial bridge) over /events (SSE) and POST /cmd.
"use strict";

const IMG_W = 320, IMG_H = 240;  // tracker image coordinates (firmware kW x kH)
const $ = (id) => document.getElementById(id);
const cam = $("cam"), ctx = cam.getContext("2d");
const plot = $("plot"), pctx = plot.getContext("2d");
const SCALE = cam.width / IMG_W;

const frame = document.createElement("canvas");  // latest camera frame at its own size
const fctx = frame.getContext("2d");
let haveFrame = false;

const rois = { L: null, R: null };     // {x0,y0,x1,y1,cal}
const curves = { L: null, R: null };   // {c:[a,b,c], sx:[...]}
let last = null;                       // last BROW values
const hist = [];                       // {t, rL, rR, nL, nR, iL, oL, iR, oR}
const HIST_SEC = 10;
let drag = null;
let connected = false;

function store(k, v) { try { localStorage.setItem("browui." + k, v); } catch (e) { /* private mode */ } }
function load(k, d) { try { const v = localStorage.getItem("browui." + k); return v === null ? d : v; } catch (e) { return d; } }

// ---------------------------------------------------------------- commands
async function send(text) {
  try {
    const r = await fetch("/cmd", { method: "POST", body: text });
    if (!r.ok) log("# " + (await r.text()));
  } catch (e) { log("# bridge not reachable"); }
}
function post(path, body) { return fetch(path, { method: "POST", body: body || "" }).catch(() => log("# bridge not reachable")); }

function initBoard() {
  send(`stream ${$("streamChk").checked ? "on" : "off"}\npreview ${$("preview").value}\nstatus\ncurve`);
}

// ---------------------------------------------------------------- console
const logEl = $("log");
const logLines = [];
function log(s) {
  logLines.push(s);
  if (logLines.length > 400) logLines.splice(0, logLines.length - 400);
  const atEnd = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 4;
  logEl.textContent = logLines.join("\n");
  if (atEnd) logEl.scrollTop = logEl.scrollHeight;
}

let curveTimer = 0;
function onLine(s) {
  log(s);
  let m;
  if ((m = s.match(/^ROI ([LR]): (set|unset)\s+\[(-?\d+) (-?\d+) (-?\d+) (-?\d+)\]\s+calibrated=(\d)/))) {
    rois[m[1]] = m[2] === "set" ? { x0: +m[3], y0: +m[4], x1: +m[5], y1: +m[6], cal: m[7] === "1" } : null;
    if (!rois[m[1]] || !rois[m[1]].cal) curves[m[1]] = null;
  } else if ((m = s.match(/^CURVE ([LR]) (.+)$/))) {
    const v = m[2].trim().split(/\s+/).map(Number);
    curves[m[1]] = { c: v.slice(0, 3), sx: v.slice(3) };
  } else if ((m = s.match(/^fps ([\d.]+)/))) {
    $("fps").textContent = `${(+m[1]).toFixed(1)} fps`;
  } else if (/^calib [LR]:|^OK roi|^OK reset/.test(s)) {
    clearTimeout(curveTimer);  // after calibration / ROI change: refresh ROIs and curves
    curveTimer = setTimeout(() => send("status\ncurve"), 300);
  } else if (/^brow_cam ready/.test(s)) {
    setTimeout(initBoard, 300);  // the board was reset: restore stream / preview
  }
}

// ---------------------------------------------------------------- events
function connect() {
  const es = new EventSource("/events");
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.t === "line") onLine(ev.s);
    else if (ev.t === "brow") onBrow(ev);
    else if (ev.t === "img") onImg(ev);
    else if (ev.t === "status") onStatus(ev);
  };
  es.onerror = () => { $("connText").textContent = "bridge not running"; $("dot").classList.remove("ok"); };
}

function onStatus(st) {
  const wasConnected = connected;
  connected = st.open;
  $("dot").classList.toggle("ok", st.open);
  $("connText").textContent = st.open ? st.port : (st.port ? `${st.port} released` : "no board found");
  $("portBtn").textContent = st.open ? "Release port" : "Take port";
  $("portBtn").dataset.open = st.open ? "1" : "";
  $("recBtn").classList.toggle("on", !!st.recording);
  $("recBtn").textContent = st.recording ? "■ Stop recording" : "● Record";
  if (st.open && !wasConnected) initBoard();
}

function onImg(ev) {
  const bin = atob(ev.data);
  if (frame.width !== ev.w || frame.height !== ev.h) { frame.width = ev.w; frame.height = ev.h; }
  const im = fctx.createImageData(ev.w, ev.h);
  for (let i = 0, j = 0; i < bin.length; ++i, j += 4) {
    const g = bin.charCodeAt(i);
    im.data[j] = im.data[j + 1] = im.data[j + 2] = g;
    im.data[j + 3] = 255;
  }
  fctx.putImageData(im, 0, 0);
  haveFrame = true;
}

function onBrow(v) {
  last = v;
  hist.push({ t: performance.now() / 1000, rL: v.rL, rR: v.rR, nL: v.nL, nR: v.nR, iL: v.iL, oL: v.oL, iR: v.iR, oR: v.oR });
  const cut = performance.now() / 1000 - HIST_SEC - 1;
  while (hist.length && hist[0].t < cut) hist.shift();
  if (v.ms !== undefined) $("ms").textContent = `tracking ${v.ms.toFixed(1)} ms/frame`;
}

// ---------------------------------------------------------------- drawing
function poly(c, x) { return (c[0] * x + c[1]) * x + c[2]; }

function drawCurve(c, xa, xb, dy) {
  ctx.beginPath();
  for (let x = xa; x <= xb; x += 1) {
    const X = x * SCALE, Y = (poly(c, x) + dy) * SCALE;
    if (x === xa) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
  }
  ctx.stroke();
}

function drawCam() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cam.width, cam.height);
  if (haveFrame) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(frame, 0, 0, cam.width, cam.height);
  } else {
    ctx.fillStyle = "#8b93a1";
    ctx.font = "14px system-ui";
    ctx.fillText($("preview").value === "0" ? "preview is off" : "waiting for the first frame…", 20, 30);
  }
  const css = getComputedStyle(document.documentElement);
  const col = { L: css.getPropertyValue("--L").trim(), R: css.getPropertyValue("--R").trim() };
  for (const s of ["L", "R"]) {
    const r = rois[s];
    if (r) {
      ctx.strokeStyle = col[s];
      ctx.lineWidth = 1.5;
      ctx.setLineDash(r.cal ? [] : [6, 4]);
      ctx.strokeRect(r.x0 * SCALE, r.y0 * SCALE, (r.x1 - r.x0) * SCALE, (r.y1 - r.y0) * SCALE);
      ctx.setLineDash([]);
      ctx.fillStyle = col[s];
      ctx.font = "bold 13px system-ui";
      ctx.fillText(s + (r.cal ? "" : " (not calibrated)"), r.x0 * SCALE + 4, r.y0 * SCALE + 15);
    }
    const cv = curves[s];
    if (cv && cv.sx.length > 1) {
      // neutral curve (green, dashed) and the tracked edge per strip (brow colour)
      ctx.strokeStyle = "rgba(60,207,122,.85)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      drawCurve(cv.c, cv.sx[0], cv.sx[cv.sx.length - 1] - 1, 0);
      ctx.setLineDash([]);
      if (last) {
        const offs = last["s" + s];
        const raise = last["r" + s];
        const valid = last["v" + s];
        ctx.strokeStyle = valid ? col[s] : "#ff5a5a";
        ctx.lineWidth = 2.5;
        for (let k = 0; k + 1 < cv.sx.length; ++k) {
          const dy = offs && offs.length > k ? offs[k] : (raise !== undefined ? -raise : 0);
          drawCurve(cv.c, cv.sx[k], cv.sx[k + 1] - 1, dy);
        }
      }
    }
  }
  if (drag) {
    const s = roiSide();
    ctx.strokeStyle = col[s];
    ctx.setLineDash([3, 3]);
    const x0 = Math.min(drag.x0, drag.x1), y0 = Math.min(drag.y0, drag.y1);
    ctx.strokeRect(x0 * SCALE, y0 * SCALE, Math.abs(drag.x1 - drag.x0) * SCALE, Math.abs(drag.y1 - drag.y0) * SCALE);
    ctx.setLineDash([]);
  }
}

function drawPlot() {
  const W = plot.width, H = plot.height;
  pctx.clearRect(0, 0, W, H);
  const norm = $("normChk").checked, ends = $("endsChk").checked;
  const now = performance.now() / 1000;
  let range = norm ? 1.5 : 10;
  if (!norm) for (const h of hist) for (const k of ["rL", "rR"]) if (h[k] !== undefined) range = Math.max(range, Math.abs(h[k]) * 1.1);
  const X = (t) => W - (now - t) / HIST_SEC * W;
  const Y = (v) => H / 2 - v / range * (H / 2 - 4);
  pctx.strokeStyle = "#333945";
  pctx.lineWidth = 1;
  pctx.beginPath();
  pctx.moveTo(0, H / 2); pctx.lineTo(W, H / 2);
  if (norm) { pctx.moveTo(0, Y(1)); pctx.lineTo(W, Y(1)); pctx.moveTo(0, Y(-1)); pctx.lineTo(W, Y(-1)); }
  pctx.stroke();
  pctx.fillStyle = "#8b93a1";
  pctx.font = "11px system-ui";
  pctx.fillText(norm ? "+1" : `+${range.toFixed(0)} px`, 4, 12);
  pctx.fillText(norm ? "−1" : `−${range.toFixed(0)} px`, 4, H - 4);
  const css = getComputedStyle(document.documentElement);
  const series = norm ? [["nL", "--L", []], ["nR", "--R", []]] : [["rL", "--L", []], ["rR", "--R", []]];
  if (ends && !norm) series.push(["iL", "--L", [2, 3]], ["oL", "--L", [6, 3]], ["iR", "--R", [2, 3]], ["oR", "--R", [6, 3]]);
  for (const [k, c, dash] of series) {
    pctx.strokeStyle = css.getPropertyValue(c).trim();
    pctx.lineWidth = dash.length ? 1 : 2;
    pctx.setLineDash(dash);
    pctx.beginPath();
    let pen = false;
    for (const h of hist) {
      const v = h[k];
      if (v === undefined || Number.isNaN(v)) { pen = false; continue; }
      if (!pen) pctx.moveTo(X(h.t), Y(v)); else pctx.lineTo(X(h.t), Y(v));
      pen = true;
    }
    pctx.stroke();
  }
  pctx.setLineDash([]);
}

function drawBars() {
  const norm = $("normChk").checked;
  $("unit").textContent = norm ? "scaled, −1 … +1" : "px (positive = up)";
  for (const s of ["L", "R"]) {
    const v = last ? last[(norm ? "n" : "r") + s] : undefined;
    const bar = $("bar" + s);
    if (v === undefined) { bar.style.width = "0"; $("val" + s).textContent = "–"; continue; }
    const f = Math.max(-1, Math.min(1, v / (norm ? 1 : 20)));
    bar.style.left = f >= 0 ? "50%" : `${50 + f * 50}%`;
    bar.style.width = `${Math.abs(f) * 50}%`;
    $("val" + s).textContent = (v >= 0 ? "+" : "") + v.toFixed(norm ? 2 : 1);
  }
}

function frameLoop() {
  drawCam();
  drawPlot();
  drawBars();
  requestAnimationFrame(frameLoop);
}

// ---------------------------------------------------------------- ROI drawing
function roiSide() { return document.querySelector("input[name=roi]:checked").value; }
function toImg(e) {
  const r = cam.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(IMG_W, Math.round((e.clientX - r.left) / r.width * IMG_W))),
    y: Math.max(0, Math.min(IMG_H, Math.round((e.clientY - r.top) / r.height * IMG_H))),
  };
}
cam.addEventListener("pointerdown", (e) => {
  const p = toImg(e);
  drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  cam.setPointerCapture(e.pointerId);
});
cam.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const p = toImg(e);
  drag.x1 = p.x; drag.y1 = p.y;
});
cam.addEventListener("pointerup", () => {
  if (!drag) return;
  const d = drag;
  drag = null;
  const x0 = Math.min(d.x0, d.x1), x1 = Math.max(d.x0, d.x1);
  const y0 = Math.min(d.y0, d.y1), y1 = Math.max(d.y0, d.y1);
  if (x1 - x0 < 16 || y1 - y0 < 16) { log("# ROI too small (min 16×16 px of 320×240)"); return; }
  const s = roiSide();
  // nose between the brows: the inner end of the image-left ROI is its right side
  let inner = s === "L" ? "right" : "left";
  if ($("swapInner").checked) inner = inner === "right" ? "left" : "right";
  send(`roi ${s} ${x0} ${y0} ${x1} ${y1} ${inner}`);
  // next drag goes to the other brow
  document.querySelector(`input[name=roi][value=${s === "L" ? "R" : "L"}]`).checked = true;
});

// ---------------------------------------------------------------- controls
$("calibBtn").onclick = () => send(`calib ${Math.max(1, Math.min(30, +$("calibN").value || 15))}`);
$("upBtn").onclick = () => send("range up");
$("downBtn").onclick = () => send("range down");
$("statusBtn").onclick = () => send("status\ncurve");
$("benchBtn").onclick = () => send("bench");
$("resetBtn").onclick = () => { if (confirm("Forget both ROIs and the calibration on the board?")) send("reset\nstatus"); };
$("streamChk").onchange = () => send(`stream ${$("streamChk").checked ? "on" : "off"}`);
$("preview").onchange = () => {
  store("preview", $("preview").value);
  if ($("preview").value === "0") haveFrame = false;
  send(`preview ${$("preview").value}`);
};
$("normChk").onchange = () => store("norm", $("normChk").checked ? "1" : "");
$("endsChk").onchange = () => store("ends", $("endsChk").checked ? "1" : "");
$("portBtn").onclick = () => post("/port", $("portBtn").dataset.open ? "close" : "open");
$("recBtn").onclick = () => post("/record", $("recBtn").classList.contains("on") ? "stop" : "start");
$("snapBtn").onclick = () => post("/snap");
$("cmdForm").onsubmit = (e) => {
  e.preventDefault();
  const c = $("cmd").value.trim();
  if (c) send(c);
  $("cmd").value = "";
};
// tracked frames per second, from the BROW lines of the last 2 s
setInterval(() => {
  const now = performance.now() / 1000;
  const n = hist.filter((h) => h.t > now - 2).length;
  if (n) $("fps").textContent = `${(n / 2).toFixed(1)} fps`;
}, 1000);

$("preview").value = load("preview", "2");
$("normChk").checked = !!load("norm", "");
$("endsChk").checked = !!load("ends", "");
connect();
requestAnimationFrame(frameLoop);
