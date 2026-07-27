/* Silk — a lit flow field, rendered as fabric and layered into depth.
 *
 * The field is still Tyler Hobbs' flow-field technique: a grid of angles driven by
 * continuous noise, curves traced through it one small step at a time. Continuity is the
 * trick — neighbouring cells differ only slightly, so adjacent curves travel almost
 * parallel and the field reads as cloth rather than as scribble.
 *
 * What makes it read as *silk* rather than as lines
 * -------------------------------------------------
 * Anisotropic strand shading, the Kajiya-Kay model — the standard for hair and fibre, and
 * the reason silk looks like silk. It shades a strand by its **tangent** rather than by a
 * surface normal, which is the only sensible choice here because a one-pixel-wide curve
 * has no meaningful normal:
 *
 *     sin(T,H) = sqrt(1 - (T·H)^2)        specular = sin(T,H)^power
 *     sin(T,L) = sqrt(1 - (T·L)^2)        diffuse  = sin(T,L)
 *
 * with T the strand tangent, L the light, V the view, and H the half-vector between them.
 * The consequence is the thing that matters: the highlight is not a dot, it is a *band*
 * running perpendicular to the fibre direction. Every strand whose tangent shares an angle
 * lights up at once, so the field develops coherent sheen bands across it — which is
 * exactly how a sheet of silk catches light, and is impossible to fake with per-strand
 * random opacity.
 *
 * Where the third dimension comes from
 * ------------------------------------
 * A second, lower-frequency noise channel is read as a height field: the sheet's folds.
 * The strand tangent is lifted out of the plane by the slope of that height along its own
 * direction, giving a genuine 3D tangent (tx, ty, tz) to shade with. So a strand running
 * up over a fold catches the light differently from one running along it, and the folds
 * appear lit rather than drawn. Height also drives line width and a small brightness lift,
 * so raised folds read as nearer.
 *
 * On top of that, three layers at different parallax rates. Far layers are dimmer, finer,
 * lower-contrast and move least. Nothing about the geometry says "distance" — the falloff
 * and the differential motion do all of it.
 *
 * Why it can afford any of this
 * -----------------------------
 * None of it is recomputed per frame. Each layer is painted once into an offscreen tile
 * taller than the viewport; a frame is three `drawImage` calls at three offsets. The
 * offsets ease toward a target derived from scroll, and that easing is what makes the
 * sheet flow rather than scroll — it keeps moving briefly after the wheel stops.
 */

/* ---- 2D simplex noise (Gustavson's public-domain formulation, seeded) --------------- */
function makeNoise(seed = 1) {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
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

const norm3 = (x, y, z) => {
  const l = Math.hypot(x, y, z) || 1;
  return [x / l, y / l, z / l];
};

const SILK_DEFAULTS = {
  seed: 7,
  curves: 620,
  steps: 400,
  stepFrac: 0.0024,     // step as a fraction of width (Hobbs: 0.001-0.005)
  scale: 0.0011,        // flow noise frequency — lower is smoother, longer features
  reliefScale: 0.00042, // fold noise frequency. Much lower: folds are large features.
  relief: 2.6,          // how far folds lift the tangent out of plane
  drift: -0.14,         // base flow angle, radians
  turn: 1.05,           // how far the noise may bend the base flow
  base: [16, 17, 22],   // unlit fibre, near-black
  sheen: [150, 176, 220], // the colour the fibre goes when it catches light
  // Light from upper-left and slightly toward the viewer. In canvas space y grows
  // downward, so a negative y is above.
  light: [-0.52, -0.66, 0.54],
  shine: 30,            // Kajiya-Kay specular exponent; higher is a tighter sheen band
  ks: 1.0,              // specular weight
  kd: 0.30,             // diffuse weight
  ambient: 0.16,
  alpha: 0.34,
  lineWidth: 0.85,
  // Sheen is quantised into this many buckets so consecutive segments sharing a bucket
  // batch into one stroke. Without it this is ~250k individual strokes and takes seconds;
  // with it the runs are long and a layer builds in tens of milliseconds. 22 is above the
  // point where banding is visible against a field this dark.
  levels: 22,
};

/** Paint one lit layer of the field into `ctx`, sized w x h in CSS pixels. */
function paintField(ctx, w, h, opts = {}) {
  const o = { ...SILK_DEFAULTS, ...opts };
  ctx.clearRect(0, 0, w, h);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // Strands are translucent and overlap; additive blending lets crossings brighten, which
  // is what gives the sheen bands their glow without a blur pass.
  ctx.globalCompositeOperation = "lighter";

  const flow = makeNoise(o.seed);
  const relief = makeNoise(o.seed * 7919 + 13);
  const step = w * o.stepFrac;

  const L = norm3(o.light[0], o.light[1], o.light[2]);
  // View is straight at the screen, so the half-vector is fixed for the whole field and
  // can be computed once rather than per segment.
  const H = norm3(L[0], L[1], L[2] + 1);

  const margin = h * 0.22;
  let v = (o.seed * 2654435761) >>> 0;
  const rnd = () => ((v = (v * 1664525 + 1013904223) >>> 0) / 4294967296);

  const heightAt = (x, y) => relief(x * o.reliefScale, y * o.reliefScale);

  for (let c = 0; c < o.curves; c++) {
    let x = -w * 0.14 + rnd() * w * 0.32;
    let y = -margin + rnd() * (h + margin * 2);

    const jitter = 0.72 + rnd() * 0.62;          // per-strand width variation
    const aBase = o.alpha * (0.55 + rnd() * 0.9);

    let runLevel = -1;
    let open = false;

    for (let i = 0; i < o.steps; i++) {
      const ang = o.drift + flow(x * o.scale, y * o.scale) * o.turn;
      const dx = Math.cos(ang) * step, dy = Math.sin(ang) * step;
      const nx = x + dx, ny = y + dy;

      if (nx < -w * 0.3 || nx > w * 1.3 || ny < -margin * 1.6 || ny > h + margin * 1.6) break;

      // Height and its slope along the direction of travel. That slope is what lifts the
      // tangent into the third dimension and makes a fold catch light.
      const hHere = heightAt(x, y);
      const tz = (heightAt(nx, ny) - hHere) * o.relief * (1 / o.stepFrac) * 0.0016;
      const T = norm3(dx, dy, tz);

      // Kajiya-Kay: the highlight is a band perpendicular to the fibre, not a point.
      const tdh = T[0] * H[0] + T[1] * H[1] + T[2] * H[2];
      const spec = Math.pow(Math.max(1 - tdh * tdh, 0), o.shine * 0.5);
      const tdl = T[0] * L[0] + T[1] * L[1] + T[2] * L[2];
      const diff = Math.sqrt(Math.max(1 - tdl * tdl, 0));

      // Folds nearer the light read brighter, which is what gives the sheet its body.
      const lift = 0.5 + 0.5 * hHere;
      let lum = o.ambient + o.kd * diff * lift + o.ks * spec;
      lum = Math.min(Math.max(lum, 0), 1.6);

      const level = Math.min((lum / 1.6 * o.levels) | 0, o.levels - 1);
      if (level !== runLevel) {
        if (open) ctx.stroke();
        const f = level / (o.levels - 1);
        // Interpolate unlit fibre toward the sheen colour, with a gamma so the lit band
        // stays narrow instead of washing the whole field pale.
        const g = f * f;
        const r = Math.round(o.base[0] + (o.sheen[0] - o.base[0]) * g);
        const gg = Math.round(o.base[1] + (o.sheen[1] - o.base[1]) * g);
        const b = Math.round(o.base[2] + (o.sheen[2] - o.base[2]) * g);
        ctx.strokeStyle = `rgba(${r},${gg},${b},${(aBase * (0.5 + 0.75 * f)).toFixed(3)})`;
        ctx.lineWidth = o.lineWidth * jitter * (0.72 + 0.85 * f) * (0.85 + 0.3 * lift);
        ctx.beginPath();
        ctx.moveTo(x, y);
        runLevel = level;
        open = true;
      }
      ctx.lineTo(nx, ny);
      x = nx; y = ny;
    }
    if (open) ctx.stroke();
  }
  ctx.globalCompositeOperation = "source-over";
}

/* ---- the hero field ------------------------------------------------------------------ */

export function mountSilk(canvas, opts = {}) {
  const draw = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintField(ctx, w, h, opts);
  };
  let t;
  draw();
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 180); });
  return draw;
}

/* ---- the page-wide sheet ------------------------------------------------------------- */

// Far to near. Each layer is a separate tile composited at its own parallax rate — the
// differential motion between them is what the eye reads as space.
const LAYERS = [
  { depth: 2, parallax: 0.16, curves: 300, scale: 0.00052, reliefScale: 0.00030,
    alpha: 0.16, lineWidth: 0.55, ks: 0.55, ambient: 0.11, overscan: 1.32, seedAdd: 0 },
  { depth: 1, parallax: 0.34, curves: 380, scale: 0.00070, reliefScale: 0.00040,
    alpha: 0.24, lineWidth: 0.75, ks: 0.85, ambient: 0.14, overscan: 1.62, seedAdd: 101 },
  { depth: 0, parallax: 0.58, curves: 440, scale: 0.00092, reliefScale: 0.00052,
    alpha: 0.34, lineWidth: 1.0, ks: 1.15, ambient: 0.16, overscan: 1.95, seedAdd: 227 },
];

const BG_DEFAULTS = {
  seed: 41,
  steps: 560,
  shine: 34,
  // Per-frame easing toward the target offset. Lower is silkier and laggier; this is the
  // number that decides whether the sheet flows or merely scrolls.
  ease: 0.075,
  sway: 22,             // px of slow horizontal drift, so it lives when nothing scrolls
  swaySpeed: 0.05,
};

/**
 * Mount the sheet as a fixed, full-viewport layer that flows with the scroll.
 * Returns a stop() for teardown.
 */
export function mountSilkBackground(canvas, opts = {}) {
  const o = { ...BG_DEFAULTS, ...opts };
  const ctx = canvas.getContext("2d");
  const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)");

  let vw = 0, vh = 0, dpr = 1, raf = 0, t0 = 0;
  let layers = [];

  const build = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    vw = window.innerWidth;
    vh = window.innerHeight;
    if (!vw || !vh) return false;

    canvas.width = Math.round(vw * dpr);
    canvas.height = Math.round(vh * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    layers = LAYERS.map((L) => {
      const tile = document.createElement("canvas");
      const tctx = tile.getContext("2d");
      const tw = vw + o.sway * 2;
      const th = Math.round(vh * L.overscan);
      tile.width = Math.round(tw * dpr);
      tile.height = Math.round(th * dpr);
      tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      paintField(tctx, tw, th, {
        ...o, ...L, seed: o.seed + L.seedAdd, steps: o.steps,
      });
      return { ...L, tile, tw, th, cur: 0, target: 0 };
    });
    return true;
  };

  const scrollFrac = () => {
    const max = Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
    return Math.min(Math.max(window.scrollY / max, 0), 1);
  };

  const retarget = () => {
    const f = scrollFrac();
    for (const L of layers) L.target = f * Math.max(L.th - vh, 0) * L.parallax;
  };

  const composite = (time) => {
    ctx.clearRect(0, 0, vw, vh);
    for (const L of layers) {
      // Nearer layers sway further, which separates them laterally as well as vertically.
      const sway = reduce?.matches ? 0
        : Math.sin(time * o.swaySpeed + L.depth * 1.1) * o.sway * (1 - L.depth * 0.28);
      ctx.drawImage(L.tile, -o.sway + sway, -L.cur, L.tw, L.th);
    }
  };

  const frame = (now) => {
    const time = (now - t0) / 1000;
    retarget();
    // Ease toward the target rather than snapping. This is the line that makes it fabric.
    for (const L of layers) L.cur += (L.target - L.cur) * o.ease;
    composite(time);
    raf = requestAnimationFrame(frame);
  };

  const start = () => {
    if (raf || reduce?.matches) return;
    t0 = performance.now();
    raf = requestAnimationFrame(frame);
  };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; };

  // Build and paint synchronously: requestAnimationFrame does not fire reliably in
  // embedded preview contexts, so anything waiting on one risks never drawing at all.
  if (!build()) return () => {};
  retarget();
  for (const L of layers) L.cur = L.target;
  composite(0);
  start();

  reduce?.addEventListener?.("change", () => {
    stop();
    if (reduce.matches) { retarget(); for (const L of layers) L.cur = L.target; composite(0); }
    else start();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop(); else start();
  });

  let rt;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = setTimeout(() => {
      if (!build()) return;
      retarget();
      for (const L of layers) L.cur = L.target;
      composite(0);
    }, 200);
  });

  return stop;
}
