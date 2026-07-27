/* Hero composition: silk flow field, the real pole-lap trace, and the glass card.
 *
 * Kept separate from app.js because it is decoration, not data. If any of it fails the
 * page still works — the hero degrades to a plain dark panel and every number below it
 * renders exactly as before.
 */

// The version query has to be repeated here, not just on the <script> tag in the HTML.
// A module's imports are fetched as their own requests with their own cache entries, so
// `hero.js?v=N` busts hero.js alone and leaves silk.js and glass.js served from cache —
// which is exactly how a rewritten background can appear not to have changed at all.
// Keep these in step with the ?v= in index.html.
import { mountSilk } from "./silk.js?v=23";
import { mountLiquidGlass } from "./glass.js?v=23";

const NS = "http://www.w3.org/2000/svg";
const mk = (t, a = {}, p = null) => {
  const n = document.createElementNS(NS, t);
  for (const [k, v] of Object.entries(a)) if (v != null) n.setAttribute(k, v);
  if (p) p.appendChild(n);
  return n;
};

const silk = document.getElementById("silk");
if (silk) mountSilk(silk, { seed: 23 });

/* The trace: Norris's actual pole lap, brightness carrying real speed. Rotated so the
   portrait-shaped circuit fills a landscape frame, and shifted right so the glass card
   never sits on top of it. */
fetch("data/trackart_2026_R11.json")
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
  .then((art) => {
    const host = document.getElementById("track-art");
    if (!host) return;
    const W = art.width, H = art.height, PAD = 60;
    const svg = mk("svg", {
      viewBox: `${-W * 0.95} ${-PAD} ${(H + PAD * 2) * 1.95} ${W + PAD * 2}`,
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": `Pole lap trace, ${art.circuit}: ${art.driver} ${art.lap_time}`,
    }, host);

    const defs = mk("defs", {}, svg);
    const glow = mk("filter", { id: "hero-glow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
    mk("feGaussianBlur", { stdDeviation: "8", result: "b" }, glow);
    const merge = mk("feMerge", {}, glow);
    mk("feMergeNode", { in: "b" }, merge);
    mk("feMergeNode", { in: "SourceGraphic" }, merge);

    const g = mk("g", { transform: `rotate(-90) translate(${-H},0)` }, svg);
    const P = art.points, S = art.speed;
    for (let i = 0; i < P.length - 1; i++) {
      const s = (S[i] + S[i + 1]) / 2;
      mk("path", {
        d: `M${P[i][0]},${P[i][1]} L${P[i + 1][0]},${P[i + 1][1]}`,
        stroke: s > 0.78 ? "#9cc0ee" : "#3d5877",
        "stroke-width": (3 + 5.5 * s).toFixed(2),
        "stroke-linecap": "round",
        fill: "none",
        opacity: (0.30 + 0.70 * Math.pow(s, 1.5)).toFixed(3),
        filter: s > 0.93 ? "url(#hero-glow)" : null,
      }, g);
    }
    const fast = P[art.fastest_ix];
    mk("circle", { cx: fast[0], cy: fast[1], r: 6, fill: "#b9d4f7", filter: "url(#hero-glow)" }, g);

    const meta = document.getElementById("race-meta");
    if (meta && meta.dataset.base) {
      meta.textContent = `${meta.dataset.base} · pole ${art.driver} ${art.lap_time}`;
    }
  })
  .catch(() => { /* no trace is fine — the silk field carries the hero on its own */ });

mountLiquidGlass(".glass", { lip: 28, ior: 1.5, scale: 36 });
