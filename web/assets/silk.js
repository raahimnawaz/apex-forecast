/* Silk — a flow-field line renderer.
 *
 * Black lines flowing on black. The method follows Tyler Hobbs' flow-field technique:
 * a grid of angles driven by continuous noise, with curves traced through it one small
 * step at a time. Continuity is the whole trick — because neighbouring grid cells differ
 * only slightly, adjacent curves travel almost parallel and the field reads as fabric
 * rather than as scribble.
 *
 * Parameters that matter, per Hobbs:
 *   step size   0.1-0.5% of width — larger than that and curves develop visible corners
 *   curve length long curves read as fluid; short ones read as fur
 *   grid margin  the field extends well beyond the canvas so curves can flow in from
 *                off-screen instead of all being born at the edge
 *
 * Canvas rather than SVG: this draws on the order of 10^5 line segments, which SVG
 * would turn into 10^5 DOM nodes. Backing store is scaled by devicePixelRatio so the
 * strands stay hairline-crisp on retina displays.
 */

/* ---- 2D simplex noise (Gustavson's public-domain formulation, seeded) --------------- */
function makeNoise(seed = 1) {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  // Deterministic shuffle so a given seed always yields the same field.
  let s = seed >>> 0 || 1;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 255; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    [p[i], p[j]] = [p[j], p[i]];
  }
  const perm = new Uint8Array(512);
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];

  const F2 = 0.5 * (Math.sqrt(3) - 1), G2 = (3 - Math.sqrt(3)) / 6;
  const grad = [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]];

  return function noise2(xin, yin) {
    const s0 = (xin + yin) * F2;
    const i = Math.floor(xin + s0), j = Math.floor(yin + s0);
    const t = (i + j) * G2;
    const x0 = xin - (i - t), y0 = yin - (j - t);
    const i1 = x0 > y0 ? 1 : 0, j1 = x0 > y0 ? 0 : 1;
    const x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2, y2 = y0 - 1 + 2 * G2;
    const ii = i & 255, jj = j & 255;
    let n = 0;
    const corner = (x, y, gi) => {
      let tt = 0.5 - x * x - y * y;
      if (tt < 0) return 0;
      tt *= tt;
      const g = grad[gi % 8];
      return tt * tt * (g[0] * x + g[1] * y);
    };
    n += corner(x0, y0, perm[ii + perm[jj]]);
    n += corner(x1, y1, perm[ii + i1 + perm[jj + j1]]);
    n += corner(x2, y2, perm[ii + 1 + perm[jj + 1]]);
    return 70 * n;
  };
}

/* ---- the renderer ------------------------------------------------------------------ */

const SILK_DEFAULTS = {
  seed: 7,
  curves: 900,          // how many strands
  steps: 420,           // length of each strand
  stepFrac: 0.0022,     // step size as a fraction of width (Hobbs: 0.001-0.005)
  scale: 0.0011,        // noise frequency — lower is smoother, longer features
  drift: -0.14,         // base flow angle in radians; slight rise across the frame
  turn: 1.05,           // how far the noise may bend the base flow
  base: [22, 22, 28],   // strand colour, near-black
  sheen: [126, 150, 196], // the rare lit strand
  sheenRate: 0.045,     // fraction of strands that catch light
  alpha: 0.30,
  lineWidth: 0.7,
};

export function renderSilk(canvas, opts = {}) {
  const o = { ...SILK_DEFAULTS, ...opts };
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;

  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // Strands are translucent and overlap; screen blending lets crossings brighten
  // slightly, which is what gives the field its sheen instead of a flat mat.
  ctx.globalCompositeOperation = "lighter";

  const noise = makeNoise(o.seed);
  const step = w * o.stepFrac;

  // Start points seeded across a band taller than the canvas, so strands flow in from
  // off-frame rather than every curve beginning on a visible edge.
  const margin = h * 0.45;
  const rndSeed = { v: o.seed * 2654435761 >>> 0 };
  const rnd = () => ((rndSeed.v = (rndSeed.v * 1664525 + 1013904223) >>> 0) / 4294967296);

  for (let c = 0; c < o.curves; c++) {
    let x = -w * 0.12 + rnd() * w * 0.30;
    let y = -margin + rnd() * (h + margin * 2);

    const lit = rnd() < o.sheenRate;
    const col = lit ? o.sheen : o.base;
    const a = lit ? o.alpha * (0.5 + rnd() * 0.9) : o.alpha * (0.45 + rnd() * 1.1);
    ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${a.toFixed(3)})`;
    ctx.lineWidth = o.lineWidth * (lit ? 1.5 : 0.75 + rnd() * 1.0);

    ctx.beginPath();
    ctx.moveTo(x, y);
    for (let i = 0; i < o.steps; i++) {
      const n = noise(x * o.scale, y * o.scale);
      const ang = o.drift + n * o.turn;
      x += Math.cos(ang) * step;
      y += Math.sin(ang) * step;
      if (x < -w * 0.3 || x > w * 1.3 || y < -margin * 1.6 || y > h + margin * 1.6) break;
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.globalCompositeOperation = "source-over";
}

/* Redraw on resize, debounced. Motion is not part of the design — the field is static,
   which keeps it out of the way of a page whose job is reading numbers. */
export function mountSilk(canvas, opts = {}) {
  let t;
  const draw = () => renderSilk(canvas, opts);
  draw();
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 180); });
  return draw;
}
