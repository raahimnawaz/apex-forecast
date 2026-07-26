/* Liquid glass — real refraction, not just a blur.
 *
 * Technique follows the published CSS/SVG approach (kube.io, "Liquid Glass in the
 * Browser", and the nikdelvin / PallavAg reference implementations): generate a
 * displacement map whose red and green channels encode how far each pixel should be
 * fetched from, then hand it to an SVG `feDisplacementMap` used as a `backdrop-filter`.
 *
 *   feImage(displacement map) -> feDisplacementMap(in=SourceGraphic, xChannel=R, yChannel=G)
 *
 * The map is built by treating the panel as a lens: thickness rises toward the edges on
 * a squircle profile, the surface normal is the numerical derivative of that profile,
 * and the ray is bent by Snell's law. Encoding is the standard one — 128 is "no shift",
 * and the `scale` attribute maps the 0-255 range onto ±scale pixels.
 *
 *   r = 128 + cos(angle) * magnitude * 127
 *   g = 128 + sin(angle) * magnitude * 127
 *
 * Browser reality: only Chromium honours an SVG filter inside `backdrop-filter`. Safari
 * and Firefox get a frosted `blur() saturate()` fallback, which still reads as glass —
 * it simply does not bend what is behind it. Support is feature-detected, never sniffed.
 */

const NS = "http://www.w3.org/2000/svg";

/** Squircle thickness profile: 0 in the flat centre, 1 at the very edge. */
function thickness(d, lip) {
  if (d >= lip) return 0;                      // flat interior — no bending here
  const t = 1 - d / lip;                       // 0 at the lip, 1 at the edge
  return Math.pow(t, 1.6);                     // squircle-ish falloff
}

/**
 * Build the displacement map for a rounded rectangle lens.
 * Returns a data URL suitable for <feImage href>.
 */
export function buildDisplacementMap(w, h, radius, {
  lip = 26,          // width of the refracting band, in px
  ior = 1.48,        // index of refraction — window glass is ~1.5
  quality = 1,
} = {}) {
  const cw = Math.max(8, Math.round(w * quality));
  const ch = Math.max(8, Math.round(h * quality));
  const c = document.createElement("canvas");
  c.width = cw; c.height = ch;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(cw, ch);
  const px = img.data;

  const sx = w / cw, sy = h / ch;
  const r = Math.min(radius, Math.min(w, h) / 2);

  // Signed distance to the rounded-rectangle edge, from the inside.
  const edgeDist = (x, y) => {
    const qx = Math.min(x, w - x), qy = Math.min(y, h - y);
    if (qx >= r && qy >= r) return Math.min(qx, qy);
    const dx = Math.max(r - qx, 0), dy = Math.max(r - qy, 0);
    return r - Math.hypot(dx, dy);
  };

  const delta = 0.75;
  for (let j = 0; j < ch; j++) {
    for (let i = 0; i < cw; i++) {
      const x = i * sx, y = j * sy;
      const d = edgeDist(x, y);
      const k = (j * cw + i) * 4;

      if (d <= 0 || d >= lip) {
        px[k] = 128; px[k + 1] = 128; px[k + 2] = 128; px[k + 3] = 255;
        continue;
      }

      // Numerical derivative of the thickness profile gives the surface normal.
      const gx = (thickness(edgeDist(x + delta, y), lip)
                - thickness(edgeDist(x - delta, y), lip)) / (2 * delta);
      const gy = (thickness(edgeDist(x, y + delta), lip)
                - thickness(edgeDist(x, y - delta), lip)) / (2 * delta);

      // Snell's law, small-angle: the bend is proportional to the surface slope.
      const bend = (ior - 1) * lip;
      let dx = -gx * bend, dy = -gy * bend;

      const mag = Math.hypot(dx, dy);
      const maxMag = bend;
      const m = Math.min(mag / maxMag, 1);
      const ang = Math.atan2(dy, dx);

      px[k]     = Math.round(128 + Math.cos(ang) * m * 127);
      px[k + 1] = Math.round(128 + Math.sin(ang) * m * 127);
      px[k + 2] = 128;
      px[k + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL();
}

/** Does this browser allow an SVG filter inside backdrop-filter? (Chromium only.) */
export function supportsBackdropSVG() {
  return CSS.supports("backdrop-filter", "url(#x)")
      || CSS.supports("-webkit-backdrop-filter", "url(#x)");
}

let uid = 0;

/**
 * Attach real refraction to an element. Falls back silently to the frosted treatment
 * the stylesheet already applies, so nothing breaks where the filter is unsupported.
 */
export function applyLiquidGlass(elm, { lip = 26, ior = 1.48, scale = 34 } = {}) {
  if (!supportsBackdropSVG()) {
    elm.dataset.glass = "frosted";
    return null;
  }
  const rect = elm.getBoundingClientRect();
  const w = Math.round(rect.width), h = Math.round(rect.height);
  if (w < 8 || h < 8) return null;

  const radius = parseFloat(getComputedStyle(elm).borderRadius) || 14;
  const href = buildDisplacementMap(w, h, radius, { lip, ior });

  const id = `lg-${++uid}`;
  let host = document.getElementById("lg-defs");
  if (!host) {
    host = document.createElementNS(NS, "svg");
    host.id = "lg-defs";
    host.setAttribute("aria-hidden", "true");
    host.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;pointer-events:none";
    document.body.appendChild(host);
  }

  const filter = document.createElementNS(NS, "filter");
  filter.setAttribute("id", id);
  filter.setAttribute("filterUnits", "userSpaceOnUse");
  filter.setAttribute("x", "0"); filter.setAttribute("y", "0");
  filter.setAttribute("width", String(w)); filter.setAttribute("height", String(h));
  filter.setAttribute("color-interpolation-filters", "sRGB");

  const feImage = document.createElementNS(NS, "feImage");
  feImage.setAttribute("href", href);
  feImage.setAttribute("x", "0"); feImage.setAttribute("y", "0");
  feImage.setAttribute("width", String(w)); feImage.setAttribute("height", String(h));
  feImage.setAttribute("result", "map");

  const disp = document.createElementNS(NS, "feDisplacementMap");
  disp.setAttribute("in", "SourceGraphic");
  disp.setAttribute("in2", "map");
  disp.setAttribute("scale", String(scale));
  disp.setAttribute("xChannelSelector", "R");
  disp.setAttribute("yChannelSelector", "G");

  filter.append(feImage, disp);
  host.appendChild(filter);

  elm.dataset.glass = "refractive";
  elm.style.backdropFilter = `url(#${id}) blur(2px) saturate(150%)`;
  elm.style.webkitBackdropFilter = `url(#${id}) blur(2px) saturate(150%)`;
  return id;
}

/** Apply to every matching element, and rebuild on resize since maps are size-specific.
 *
 * Sizing is the fiddly part: a displacement map is generated for one exact pixel size, so
 * measuring before layout settles produces a zero-size element and silently no glass.
 * Rather than guess at a delay, this observes each element and (re)builds whenever its
 * box actually changes — which also covers font loading and late-arriving content.
 */
export function mountLiquidGlass(selector, opts = {}) {
  const build = (e) => {
    if (e.dataset.glassId) document.getElementById(e.dataset.glassId)?.remove();
    const id = applyLiquidGlass(e, opts);
    if (id) e.dataset.glassId = id;
    return id;
  };

  const seen = new WeakMap();
  const sized = (e) => {
    const r = e.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return false;
    const key = `${Math.round(r.width)}x${Math.round(r.height)}`;
    if (seen.get(e) === key) return false;
    seen.set(e, key);
    return true;
  };

  // Build synchronously first. A deferred module runs after layout, so the boxes are
  // already measurable — and frame-driven callbacks (rAF, ResizeObserver) are not
  // guaranteed to fire in every embedding context, so the first paint must not depend
  // on one. ResizeObserver then handles later changes: resize, font swap, late content.
  const buildAll = () => document.querySelectorAll(selector).forEach((e) => {
    if (sized(e)) build(e);
  });
  buildAll();

  if (typeof ResizeObserver === "function") {
    const ro = new ResizeObserver((entries) => {
      for (const { target } of entries) if (sized(target)) build(target);
    });
    document.querySelectorAll(selector).forEach((e) => ro.observe(e));
  }
  return buildAll;
}
