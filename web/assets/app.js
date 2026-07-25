/* apex-forecast dashboard renderer.
   Hand-rolled SVG: no chart library, no CDN, no build step. The page loads one JSON
   payload and nothing else, so it deploys to any static host. */

const SVG = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent = null) => {
  const n = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const fmtAbs = (v, d = 3) => v.toFixed(d);

/* ---------- tooltip ---------- */

const tip = document.getElementById("tip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const r = evt.target.getBoundingClientRect();
  tip.style.left = `${r.left + r.width / 2}px`;
  tip.style.top = `${r.top}px`;
}
function hideTip() { tip.classList.remove("on"); }

function tipRow(label, value) {
  return `<div class="t-row"><span>${label}</span><b>${value}</b></div>`;
}

/* Attach hover + keyboard focus to a mark. Tooltips enhance; every value also
   lives in the table view, so nothing is gated behind a pointer. */
function interactive(node, html) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("mouseenter", (e) => showTip(e, html));
  node.addEventListener("focus", (e) => showTip(e, html));
  node.addEventListener("mouseleave", hideTip);
  node.addEventListener("blur", hideTip);
}

/* A bar with its data-end rounded and its baseline-end square. */
function barPath(x0, x1, y, h, r = 4) {
  const right = x1 >= x0;
  const rr = Math.min(r, Math.abs(x1 - x0));
  return right
    ? `M${x0},${y} H${x1 - rr} Q${x1},${y} ${x1},${y + rr} V${y + h - rr} Q${x1},${y + h} ${x1 - rr},${y + h} H${x0} Z`
    : `M${x0},${y} H${x1 + rr} Q${x1},${y} ${x1},${y + rr} V${y + h - rr} Q${x1},${y + h} ${x1 + rr},${y + h} H${x0} Z`;
}

/* ---------- 1. corrected race pace ---------- */

function renderPace(data) {
  const rows = data.pace;
  const host = document.getElementById("pace-chart");
  const ROW = 27, BAR = 14, PAD_T = 26, PAD_B = 34, GUTTER = 132, PAD_R = 62;
  const W = 860, H = PAD_T + rows.length * ROW + PAD_B;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Fuel- and tyre-corrected race pace by driver, seconds relative to field mean" }, host);

  const lo = Math.min(...rows.map(r => r.pace_s - r.se_s));
  const hi = Math.max(...rows.map(r => r.pace_s + r.se_s));
  const span = Math.max(Math.abs(lo), Math.abs(hi)) * 1.12;
  const x = (v) => GUTTER + ((v + span) / (2 * span)) * (W - GUTTER - PAD_R);
  const x0 = x(0);

  // gridlines + ticks
  const step = span > 1.2 ? 0.5 : 0.25;
  for (let t = -Math.ceil(span / step) * step; t <= span; t += step) {
    if (Math.abs(t) < 1e-9) continue;
    if (x(t) < GUTTER || x(t) > W - PAD_R + 1) continue;
    el("line", { class: "gridline", x1: x(t), x2: x(t), y1: PAD_T - 8, y2: H - PAD_B }, svg);
    const lb = el("text", { class: "tick-label", x: x(t), y: H - PAD_B + 16, "text-anchor": "middle" }, svg);
    lb.textContent = fmt(t, 2);
  }
  el("line", { class: "zero-line", x1: x0, x2: x0, y1: PAD_T - 8, y2: H - PAD_B }, svg);
  const zl = el("text", { x: x0, y: PAD_T - 14, "text-anchor": "middle" }, svg);
  zl.textContent = "field mean";

  const faster = el("text", { x: GUTTER, y: H - PAD_B + 32, "text-anchor": "start" }, svg);
  faster.textContent = "← faster (s/lap)";
  const slower = el("text", { x: W - PAD_R, y: H - PAD_B + 32, "text-anchor": "end" }, svg);
  slower.textContent = "slower (s/lap) →";

  rows.forEach((r, i) => {
    const y = PAD_T + i * ROW;
    const yb = y + (ROW - BAR) / 2;

    // team identity chip — identity only, never the quantitative scale
    el("rect", { x: 0, y: yb + 3, width: 8, height: 8, rx: 2,
      fill: data.team_colors[r.Team] || css("--ink-muted") }, svg);
    const nm = el("text", { class: "name-label" + (i < 3 ? " lead" : ""), x: 14, y: yb + BAR - 3 }, svg);
    nm.textContent = r.Driver;
    const tm = el("text", { x: 58, y: yb + BAR - 3 }, svg);
    tm.textContent = r.Team.replace(" F1 Team", "").replace(" Racing", "");

    el("path", { d: barPath(x0, x(r.pace_s), yb, BAR), fill: css("--series-1") }, svg);

    // ±1 standard error of the across-race mean
    const eL = x(r.pace_s - r.se_s), eR = x(r.pace_s + r.se_s), yc = yb + BAR / 2;
    el("line", { class: "err", x1: eL, x2: eR, y1: yc, y2: yc }, svg);
    el("line", { class: "err", x1: eL, x2: eL, y1: yc - 3.5, y2: yc + 3.5 }, svg);
    el("line", { class: "err", x1: eR, x2: eR, y1: yc - 3.5, y2: yc + 3.5 }, svg);

    // label the extremes only — a number on every bar goes unread
    if (i < 3 || i === rows.length - 1) {
      const right = r.pace_s >= 0;
      const lx = right ? Math.max(eR, x(r.pace_s)) + 8 : Math.min(eL, x(r.pace_s)) - 8;
      const vl = el("text", { class: "val-label", x: lx, y: yb + BAR - 3,
        "text-anchor": right ? "start" : "end" }, svg);
      vl.textContent = fmt(r.pace_s);
    }

    const hit = el("rect", { class: "hit", x: 0, y, width: W, height: ROW }, svg);
    interactive(hit, `<div class="t-name"><span class="chip" style="background:${data.team_colors[r.Team]}"></span>${r.Driver}</div>`
      + tipRow("Team", r.Team)
      + tipRow("Corrected pace", `${fmt(r.pace_s)} s`)
      + tipRow("Std. error", `± ${fmtAbs(r.se_s)} s`)
      + tipRow("Race-to-race SD", `± ${fmtAbs(r.sd_s)} s`)
      + tipRow("Races / laps", `${r.races} / ${r.laps}`));
  });

  // table twin
  const tb = document.querySelector("#pace-table tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.rank}</td><td class="name">${r.Driver}</td><td class="name">${r.Team}</td>`
      + `<td>${fmt(r.pace_s)}</td><td>${fmtAbs(r.se_s)}</td><td>${fmtAbs(r.sd_s)}</td>`
      + `<td>${r.races}</td><td>${r.laps}</td>`;
    tb.appendChild(tr);
  });
}

/* ---------- 2. tyre degradation ---------- */

function renderDeg(data) {
  const comps = data.degradation.filter(d => ["SOFT", "MEDIUM", "HARD"].includes(d.compound));
  const host = document.getElementById("deg-chart");
  const W = 560, H = 300, L = 46, R = 74, T = 18, B = 42;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Modelled cumulative time loss versus tyre age, by compound" }, host);

  const MAXAGE = 30;
  const maxLoss = Math.max(0.6, ...comps.map(c => c.deg_s_per_lap * MAXAGE)) * 1.1;
  const x = (a) => L + (a / MAXAGE) * (W - L - R);
  const y = (v) => H - B - (v / maxLoss) * (H - T - B);

  // ordinal ramp: compound hardness is an ordered category, so darkness carries order
  const ramp = { SOFT: css("--seq-250"), MEDIUM: css("--seq-400"), HARD: css("--seq-600") };

  for (let v = 0; v <= maxLoss; v += maxLoss / 4) {
    el("line", { class: "gridline", x1: L, x2: W - R, y1: y(v), y2: y(v) }, svg);
    const t = el("text", { class: "tick-label", x: L - 8, y: y(v) + 4, "text-anchor": "end" }, svg);
    t.textContent = v.toFixed(1);
  }
  el("line", { class: "axis-line", x1: L, x2: W - R, y1: y(0), y2: y(0) }, svg);
  for (let a = 0; a <= MAXAGE; a += 10) {
    const t = el("text", { class: "tick-label", x: x(a), y: H - B + 17, "text-anchor": "middle" }, svg);
    t.textContent = a;
  }
  const xl = el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }, svg);
  xl.textContent = "tyre age (laps)";
  const yl = el("text", { x: -(H / 2), y: 12, "text-anchor": "middle", transform: "rotate(-90)" }, svg);
  yl.textContent = "cumulative loss (s)";

  comps.forEach((c) => {
    const col = ramp[c.compound];
    const yEnd = y(c.deg_s_per_lap * MAXAGE);
    el("line", { x1: x(0), y1: y(0), x2: x(MAXAGE), y2: yEnd,
      stroke: col, "stroke-width": 2, "stroke-linecap": "round" }, svg);
    // end marker with a 2px surface ring so overlapping ends stay legible
    el("circle", { cx: x(MAXAGE), cy: yEnd, r: 4.5, fill: col,
      stroke: css("--surface-1"), "stroke-width": 2 }, svg);
    const lb = el("text", { class: "val-label", x: x(MAXAGE) + 10, y: yEnd + 4 }, svg);
    lb.textContent = c.compound[0] + c.compound.slice(1).toLowerCase();

    const hit = el("rect", { class: "hit", x: L, y: Math.min(yEnd, y(0)) - 9,
      width: W - L - R, height: 18 }, svg);
    interactive(hit, `<div class="t-name">${c.compound}</div>`
      + tipRow("Degradation", `${fmtAbs(c.deg_s_per_lap, 4)} s/lap`)
      + tipRow("Across races SD", `± ${fmtAbs(c.sd, 4)}`)
      + tipRow("Loss at 30 laps", `${fmtAbs(c.deg_s_per_lap * 30, 2)} s`)
      + tipRow("Races / laps", `${c.races} / ${c.laps}`));
  });

  const lg = document.getElementById("deg-legend");
  comps.forEach((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="swatch line" style="background:${ramp[c.compound]}"></span>`
      + `${c.compound[0]}${c.compound.slice(1).toLowerCase()}`;
    lg.appendChild(li);
  });

  const tb = document.querySelector("#deg-table tbody");
  data.degradation.forEach((c) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="name">${c.compound}</td><td>${fmtAbs(c.deg_s_per_lap, 4)}</td>`
      + `<td>${fmtAbs(c.sd, 4)}</td><td>${fmtAbs(c.deg_s_per_lap * 30, 2)}</td>`
      + `<td>${c.races}</td><td>${c.laps}</td>`;
    tb.appendChild(tr);
  });
}

/* ---------- 3. pace by round (diverging heatmap) ---------- */

function renderForm(data) {
  const order = data.pace.map(p => p.Driver);
  const rounds = data.rounds_analysed;
  const idx = new Map();
  data.pace_by_round.forEach(r => idx.set(`${r.Driver}|${r.round}`, r));
  const eventOf = new Map(data.pace_by_round.map(r => [r.round, r.event]));

  const host = document.getElementById("form-chart");
  const CELL = 26, GAPC = 2, ROW = 22, GUTTER = 58, T = 30, B = 12;
  const W = GUTTER + rounds.length * (CELL + GAPC) + 8;
  const H = T + order.length * ROW + B;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`,
    style: `min-width:${W}px`, role: "img",
    "aria-label": "Corrected pace by driver and round, relative to the field mean in each race" }, host);

  const vals = data.pace_by_round.map(r => Math.abs(r.pace_s));
  const cap = Math.max(0.4, ...vals) * 0.92;

  // diverging: colour must carry the sign here, because a heatmap cell has no
  // position channel to spend on it (unlike the bars above)
  const cool = css("--div-cool"), warm = css("--div-warm"), mid = css("--div-mid");
  const mix = (a, b, t) => {
    const p = (h) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    const [r1, g1, b1] = p(a), [r2, g2, b2] = p(b);
    const c = (u, v) => Math.round(u + (v - u) * t);
    return `rgb(${c(r1, r2)},${c(g1, g2)},${c(b1, b2)})`;
  };
  const color = (v) => {
    const t = Math.min(1, Math.abs(v) / cap);
    return v < 0 ? mix(mid, cool, t) : mix(mid, warm, t);
  };

  rounds.forEach((rd, j) => {
    const t = el("text", { class: "tick-label", x: GUTTER + j * (CELL + GAPC) + CELL / 2,
      y: T - 12, "text-anchor": "middle" }, svg);
    t.textContent = "R" + rd;
  });

  order.forEach((drv, i) => {
    const y = T + i * ROW;
    const nm = el("text", { class: "name-label", x: 0, y: y + ROW - 7 }, svg);
    nm.textContent = drv;
    rounds.forEach((rd, j) => {
      const rec = idx.get(`${drv}|${rd}`);
      const cx = GUTTER + j * (CELL + GAPC);
      if (!rec) {
        // "No data" must not be confusable with "at the field mean": the diverging
        // midpoint sits at 1.48:1 against the surface, so a mid cell is deliberately
        // faint. A 45° hairline marks absence as a state rather than a value.
        el("line", { x1: cx + 6, y1: y + ROW - 7, x2: cx + CELL - 6, y2: y + 7,
          stroke: css("--axis"), "stroke-width": 1 }, svg);
        return;
      }
      const cell = el("rect", { x: cx, y: y + 2, width: CELL, height: ROW - 4, rx: 3,
        fill: color(rec.pace_s) }, svg);
      interactive(cell, `<div class="t-name">${drv} · R${rd}</div>`
        + tipRow("Race", eventOf.get(rd) || "")
        + tipRow("Corrected pace", `${fmt(rec.pace_s)} s`)
        + tipRow("Laps used", rec.n_laps));
    });
  });

  // scale legend
  const lg = document.getElementById("form-legend");
  lg.innerHTML = `<li><span class="swatch" style="background:${mix(mid, cool, 1)}"></span>faster than field mean</li>`
    + `<li><span class="swatch" style="background:${mid}"></span>at field mean</li>`
    + `<li><span class="swatch" style="background:${mix(mid, warm, 1)}"></span>slower than field mean</li>`
    + `<li><span class="swatch" style="background:linear-gradient(45deg,transparent 44%,${css("--axis")} 44%,${css("--axis")} 56%,transparent 56%)"></span>no valid race laps</li>`;

  const head = document.querySelector("#form-table thead tr");
  head.innerHTML = "<th>Driver</th>" + rounds.map(r => `<th>R${r}</th>`).join("");
  const tb = document.querySelector("#form-table tbody");
  order.forEach((drv) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="name">${drv}</td>`
      + rounds.map(rd => {
        const rec = idx.get(`${drv}|${rd}`);
        return `<td>${rec ? fmt(rec.pace_s, 2) : "—"}</td>`;
      }).join("");
    tb.appendChild(tr);
  });
}

/* ---------- header, tiles, notes ---------- */

function renderHeader(data) {
  const ne = data.next_event;
  document.getElementById("race-name").textContent = ne.name;
  document.getElementById("race-meta").textContent =
    `Round ${ne.round} of ${ne.total_rounds} · ${ne.location}, ${ne.country} · ${ne.date}`
    + (ne.format.includes("sprint") ? " · sprint weekend" : "");

  // Calendar-day difference, not elapsed hours: "days away" counts dates, so an
  // event tomorrow reads 1 whatever the local time of day is now.
  const [yy, mm, dd] = ne.date.split("-").map(Number);
  const raceDay = Date.UTC(yy, mm - 1, dd);
  const now = new Date();
  const today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  const days = Math.max(0, Math.round((raceDay - today) / 86400000));
  document.getElementById("cd-num").textContent = days === 0 ? "Today" : days;
  document.getElementById("cd-unit").textContent =
    days === 0 ? "race day" : days === 1 ? "day away" : "days away";

  document.getElementById("stamp").textContent = "built " + data.generated_utc.replace("T", " ");

  const t = data.totals;
  const lead = data.pace[0];
  const tiles = [
    ["Races deconvolved", t.races_fitted, `of ${data.rounds_analysed.length} completed rounds`],
    ["Green-flag laps modelled", t.laps_modelled.toLocaleString(), "after fuel, tyre and traffic correction"],
    ["Quickest corrected pace", fmt(lead.pace_s) + " s", `${lead.Driver} · ${lead.Team}`],
    ["Cost of dirty air", fmtAbs(t.mean_dirty_air_cost_s, 2) + " s", "at zero gap, mean across races"],
  ];
  const host = document.getElementById("tiles");
  tiles.forEach(([label, value, sub]) => {
    const d = document.createElement("div");
    d.className = "tile";
    d.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div>`;
    host.appendChild(d);
  });

  document.getElementById("fit-quality").textContent =
    `Median pseudo-R² ${t.median_pseudo_r2.toFixed(3)} across ${t.races_fitted} races; `
    + `residual SD ${t.mean_resid_sd_s.toFixed(3)} s.`;
}

/* ---------- boot ---------- */

fetch("data/pace_2026.json")
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
  .then((data) => {
    renderHeader(data);
    renderPace(data);
    renderDeg(data);
    renderForm(data);
  })
  .catch((e) => {
    document.getElementById("tiles").innerHTML =
      `<div class="tile"><div class="label">Data</div><div class="value">—</div>`
      + `<div class="sub">Could not load data/pace_2026.json (${e.message}). `
      + `Run <code>scripts/export_web.py</code> first.</div></div>`;
  });
