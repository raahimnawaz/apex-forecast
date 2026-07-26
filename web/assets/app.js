/* apex-forecast dashboard renderer.
   Hand-rolled SVG: no chart library, no CDN, no build step. The page loads a handful of
   JSON payloads and nothing else, so it deploys to any static host.

   Security note: everything under `news` originates from third-party RSS feeds. It is
   treated strictly as untrusted text — inserted with textContent, never innerHTML, never
   evaluated, and never interpreted as an instruction. */

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
/* Below the sampling floor a probability is not "zero", it is "unresolved". These come
   out of a finite Monte Carlo, so an empty bucket means "rarer than we sampled", not
   "impossible" — and no outcome here is impossible. Printing 0.0% would claim a
   certainty the model has not earned. */
const pct = (v, d = 1) => (v < 0.001 ? "<0.1%" : (100 * v).toFixed(d) + "%");
const shortTeam = (t) => t.replace(" F1 Team", "").replace(" Racing", "");

const state = { selectedTeam: null, newsFilter: null };

/* ---------- tooltip ---------- */

const tip = document.getElementById("tip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const r = evt.target.getBoundingClientRect();
  tip.style.left = `${r.left + r.width / 2}px`;
  tip.style.top = `${r.top}px`;
}
const hideTip = () => tip.classList.remove("on");
const tipRow = (label, value) => `<div class="t-row"><span>${label}</span><b>${value}</b></div>`;

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

const mixHex = (a, b, t) => {
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [r1, g1, b1] = p(a), [r2, g2, b2] = p(b);
  const c = (u, v) => Math.round(u + (v - u) * t);
  return `rgb(${c(r1, r2)},${c(g1, g2)},${c(b1, b2)})`;
};

/* =================== FORECAST =================== */

function renderForecast(s, colors) {
  const rows = [...s.forecast].sort((a, b) => b.p_win - a.p_win);
  const host = document.getElementById("win-chart");
  const ROW = 27, BAR = 14, PAD_T = 12, PAD_B = 36, GUTTER = 132, PAD_R = 58;
  const W = 860, H = PAD_T + rows.length * ROW + PAD_B;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Win probability by driver for the next race" }, host);

  const max = Math.max(...rows.map((r) => r.p_win)) * 1.16;
  const x = (v) => GUTTER + (v / max) * (W - GUTTER - PAD_R);

  const step = max > 0.25 ? 0.05 : 0.02;
  for (let t = step; t <= max; t += step) {
    el("line", { class: "gridline", x1: x(t), x2: x(t), y1: PAD_T - 4, y2: H - PAD_B }, svg);
    const lb = el("text", { class: "tick-label", x: x(t), y: H - PAD_B + 16, "text-anchor": "middle" }, svg);
    lb.textContent = Math.round(100 * t) + "%";
  }
  el("line", { class: "axis-line", x1: GUTTER, x2: GUTTER, y1: PAD_T - 4, y2: H - PAD_B }, svg);
  const xl = el("text", { x: GUTTER, y: H - PAD_B + 32 }, svg);
  xl.textContent = "probability of winning, conditional on finishing";

  rows.forEach((r, i) => {
    const y = PAD_T + i * ROW, yb = y + (ROW - BAR) / 2;
    el("rect", { x: 0, y: yb + 3, width: 8, height: 8, rx: 2,
      fill: colors[r.team] || css("--ink-muted") }, svg);
    const nm = el("text", { class: "name-label" + (i < 3 ? " lead" : ""), x: 14, y: yb + BAR - 3 }, svg);
    nm.textContent = r.driver;
    const tm = el("text", { x: 58, y: yb + BAR - 3 }, svg);
    tm.textContent = shortTeam(r.team);

    el("path", { d: barPath(x(0), x(r.p_win), yb, BAR), fill: css("--series-1") }, svg);
    if (i < 4) {
      const vl = el("text", { class: "val-label", x: x(r.p_win) + 8, y: yb + BAR - 3 }, svg);
      vl.textContent = pct(r.p_win);
    }

    const hit = el("rect", { class: "hit", x: 0, y, width: W, height: ROW }, svg);
    interactive(hit, `<div class="t-name"><span class="chip" style="background:${colors[r.team]}"></span>${r.driver}</div>`
      + tipRow("Team", r.team)
      + tipRow("Win", pct(r.p_win))
      + tipRow("Podium", pct(r.p_podium))
      + tipRow("Points", pct(r.p_points))
      + tipRow("Expected finish", "P" + r.exp_pos.toFixed(1)));
  });

  const tb = document.querySelector("#forecast-table tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="name">${r.driver}</td><td class="name">${r.team}</td>`
      + `<td>${r.grid ?? "—"}</td>`
      + `<td>${pct(r.p_win)}</td><td>${pct(r.p_podium)}</td><td>${pct(r.p_points)}</td>`
      + `<td>${r.exp_pos.toFixed(1)}</td>`;
    tb.appendChild(tr);
  });

}

/* =================== POSITION MATRIX =================== */

// Sequential bins. On a dark surface the low end recedes toward the surface and the
// high end is the brightest step, so "more likely" reads as "brighter".
const MATRIX_BINS = [
  { min: 0.35, hex: "#9ec5f4", label: "≥ 35%" },
  { min: 0.20, hex: "#6da7ec", label: "20–35%" },
  { min: 0.10, hex: "#3987e5", label: "10–20%" },
  { min: 0.05, hex: "#256abf", label: "5–10%" },
  { min: 0.02, hex: "#184f95", label: "2–5%" },
  { min: 0.005, hex: "#0d366b", label: "0.5–2%" },
];
const binColor = (p) => (MATRIX_BINS.find((b) => p >= b.min) || {}).hex || null;

function renderMatrix(s, colors) {
  const { drivers, teams, probs } = s.position_matrix;
  const order = [...drivers.keys()].sort((a, b) =>
    probs[a].reduce((m, p, i) => m + p * i, 0) - probs[b].reduce((m, p, i) => m + p * i, 0));

  const host = document.getElementById("matrix-chart");
  const N = drivers.length, CELL = 25, GAPC = 2, ROW = 22, GUTTER = 118, T = 32, B = 26;
  const W = GUTTER + N * (CELL + GAPC) + 8, H = T + N * ROW + B;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, style: `min-width:${W}px`,
    role: "img", "aria-label": "Probability of each finishing position, by driver" }, host);

  for (let p = 0; p < N; p++) {
    if (p % 2 === 0 || p === N - 1) {
      const t = el("text", { class: "tick-label", x: GUTTER + p * (CELL + GAPC) + CELL / 2,
        y: T - 12, "text-anchor": "middle" }, svg);
      t.textContent = "P" + (p + 1);
    }
  }

  order.forEach((di, i) => {
    const y = T + i * ROW;
    el("rect", { x: 0, y: y + (ROW - 8) / 2, width: 8, height: 8, rx: 2,
      fill: colors[teams[di]] || css("--ink-muted") }, svg);
    const nm = el("text", { class: "name-label", x: 14, y: y + ROW - 7 }, svg);
    nm.textContent = drivers[di];
    const tm = el("text", { x: 56, y: y + ROW - 7, "font-size": "10.5" }, svg);
    tm.textContent = shortTeam(teams[di]).slice(0, 9);

    for (let p = 0; p < N; p++) {
      const prob = probs[di][p];
      const fill = binColor(prob);
      const cx = GUTTER + p * (CELL + GAPC);
      if (!fill) continue;
      const cell = el("rect", { x: cx, y: y + 2, width: CELL, height: ROW - 4, rx: 3, fill }, svg);
      interactive(cell, `<div class="t-name">${drivers[di]} · P${p + 1}</div>`
        + tipRow("Probability", pct(prob, 1))
        + tipRow("Team", teams[di]));
    }
  });

  const lg = document.getElementById("matrix-legend");
  lg.innerHTML = MATRIX_BINS.map((b) =>
    `<li><span class="swatch" style="background:${b.hex}"></span>${b.label}</li>`).join("")
    + `<li><span class="swatch" style="background:transparent;border:1px solid ${css("--grid")}"></span>&lt; 0.5%</li>`;

  const head = document.querySelector("#matrix-table thead tr");
  head.innerHTML = "<th>Driver</th>" + Array.from({ length: N }, (_, p) => `<th>P${p + 1}</th>`).join("");
  const tb = document.querySelector("#matrix-table tbody");
  order.forEach((di) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="name">${drivers[di]}</td>`
      + probs[di].map((p) => `<td>${p >= 0.005 ? (100 * p).toFixed(1) : "—"}</td>`).join("");
    tb.appendChild(tr);
  });
}

/* =================== LAYER 1 SPLIT =================== */

function ciBarChart(hostId, rows, opts) {
  const host = document.getElementById(hostId);
  const ROW = 24, BAR = 12, PAD_T = 18, PAD_B = 34, GUTTER = opts.gutter || 92, PAD_R = 16;
  const W = 460, H = PAD_T + rows.length * ROW + PAD_B;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": opts.label }, host);

  const lo = Math.min(...rows.map((r) => r.lo)), hi = Math.max(...rows.map((r) => r.hi));
  const span = Math.max(Math.abs(lo), Math.abs(hi)) * 1.08;
  const x = (v) => GUTTER + ((v + span) / (2 * span)) * (W - GUTTER - PAD_R);
  const x0 = x(0);

  for (let t = -Math.floor(span); t <= span; t += 1) {
    if (Math.abs(t) < 1e-9) continue;
    el("line", { class: "gridline", x1: x(t), x2: x(t), y1: PAD_T - 6, y2: H - PAD_B }, svg);
    const lb = el("text", { class: "tick-label", x: x(t), y: H - PAD_B + 15, "text-anchor": "middle" }, svg);
    lb.textContent = t;
  }
  el("line", { class: "zero-line", x1: x0, x2: x0, y1: PAD_T - 6, y2: H - PAD_B }, svg);
  const al = el("text", { x: (GUTTER + W - PAD_R) / 2, y: H - PAD_B + 30, "text-anchor": "middle" }, svg);
  al.textContent = opts.axis;

  rows.forEach((r, i) => {
    const y = PAD_T + i * ROW, yb = y + (ROW - BAR) / 2, yc = yb + BAR / 2;
    if (r.color) el("rect", { x: 0, y: yb + 2, width: 8, height: 8, rx: 2, fill: r.color }, svg);
    const nm = el("text", { class: "name-label", x: r.color ? 14 : 0, y: yb + BAR - 2 }, svg);
    nm.textContent = r.name;

    el("path", { d: barPath(x0, x(r.value), yb, BAR), fill: css("--series-1") }, svg);
    el("line", { class: "err", x1: x(r.lo), x2: x(r.hi), y1: yc, y2: yc }, svg);
    el("line", { class: "err", x1: x(r.lo), x2: x(r.lo), y1: yc - 3, y2: yc + 3 }, svg);
    el("line", { class: "err", x1: x(r.hi), x2: x(r.hi), y1: yc - 3, y2: yc + 3 }, svg);

    const hit = el("rect", { class: "hit", x: 0, y, width: W, height: ROW }, svg);
    interactive(hit, `<div class="t-name">${r.name}</div>`
      + tipRow(opts.valueLabel, fmt(r.value, 2))
      + tipRow("89% interval", `${fmt(r.lo, 2)} … ${fmt(r.hi, 2)}`)
      + (r.extra || ""));
  });
}

function renderSplit(s, colors, gridDrivers) {
  const skill = s.skill.filter((d) => gridDrivers.has(d.driver));
  ciBarChart("skill-chart", skill.map((d) => ({
    name: d.driver, value: d.skill, lo: d.lo, hi: d.hi,
  })), { label: "Driver skill with 89% credible intervals", axis: "skill (log-odds)",
         valueLabel: "Skill", gutter: 46 });

  const cons = [...s.constructor].sort((a, b) => b.car_2026_latest - a.car_2026_latest);
  ciBarChart("cons-chart", cons.map((c) => ({
    name: shortTeam(c.constructor), value: c.car_2026_latest, lo: c.lo, hi: c.hi,
    color: colors[c.constructor],
    extra: tipRow("Development since R1", fmt(c.development, 2)),
  })), { label: "Constructor strength with 89% credible intervals",
         axis: "car strength (log-odds)", valueLabel: "Strength", gutter: 108 });

  const tb = document.querySelector("#skill-table tbody");
  skill.forEach((d) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="name">${d.driver}</td><td>${fmt(d.skill, 3)}</td>`
      + `<td>${fmt(d.lo, 3)}</td><td>${fmt(d.hi, 3)}</td>`;
    tb.appendChild(tr);
  });
}

/* =================== DEVELOPMENT =================== */

let devRedraw = null;

function renderDev(s, colors) {
  const host = document.getElementById("dev-chart");
  const byTeam = new Map();
  s.constructor_by_round.forEach((r) => {
    if (!byTeam.has(r.constructor)) byTeam.set(r.constructor, []);
    byTeam.get(r.constructor).push(r);
  });
  byTeam.forEach((v) => v.sort((a, b) => a.round - b.round));

  const rounds = [...new Set(s.constructor_by_round.map((r) => r.round))].sort((a, b) => a - b);
  const W = 860, H = 340, L = 48, R = 118, T = 18, B = 44;
  const all = s.constructor_by_round;
  const lo = Math.min(...all.map((r) => r.lo)), hi = Math.max(...all.map((r) => r.hi));
  const x = (r) => L + ((r - rounds[0]) / (rounds.at(-1) - rounds[0])) * (W - L - R);
  const y = (v) => H - B - ((v - lo) / (hi - lo)) * (H - T - B);

  function draw() {
    host.innerHTML = "";
    const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Constructor strength by round through the 2026 season" }, host);

    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      el("line", { class: "gridline", x1: L, x2: W - R, y1: y(v), y2: y(v) }, svg);
      const t = el("text", { class: "tick-label", x: L - 8, y: y(v) + 4, "text-anchor": "end" }, svg);
      t.textContent = v.toFixed(1);
    }
    rounds.forEach((r) => {
      const t = el("text", { class: "tick-label", x: x(r), y: H - B + 17, "text-anchor": "middle" }, svg);
      t.textContent = "R" + r;
    });
    const xl = el("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle" }, svg);
    xl.textContent = "round";
    const yl = el("text", { x: -(H / 2), y: 12, "text-anchor": "middle", transform: "rotate(-90)" }, svg);
    yl.textContent = "car strength (log-odds)";

    const sel = state.selectedTeam;

    // Emphasis rather than eleven hues: past eight categorical slots the manual says
    // fold or highlight, so every team is drawn recessive and the selected one leads.
    if (sel && byTeam.has(sel)) {
      const pts = byTeam.get(sel);
      const band = pts.map((p) => `${x(p.round)},${y(p.hi)}`).join(" ")
        + " " + [...pts].reverse().map((p) => `${x(p.round)},${y(p.lo)}`).join(" ");
      el("polygon", { points: band, fill: css("--series-1"), opacity: 0.1 }, svg);
    }

    byTeam.forEach((pts, team) => {
      const isSel = team === sel;
      const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.round)},${y(p.strength)}`).join(" ");
      el("path", { d, fill: "none",
        stroke: isSel ? css("--series-1") : css("--axis"),
        "stroke-width": isSel ? 2 : 1.25,
        "stroke-linecap": "round", "stroke-linejoin": "round",
        opacity: sel ? (isSel ? 1 : 0.45) : 0.7 }, svg);

      const last = pts.at(-1);
      if (isSel || !sel) {
        el("circle", { cx: x(last.round), cy: y(last.strength), r: isSel ? 4.5 : 3,
          fill: isSel ? css("--series-1") : css("--axis"),
          stroke: css("--surface-1"), "stroke-width": 2 }, svg);
      }
      const lab = el("text", { class: isSel ? "val-label" : "", x: x(last.round) + 9,
        y: y(last.strength) + 4, "font-size": isSel ? "12" : "10.5",
        opacity: sel && !isSel ? 0.5 : 1 }, svg);
      lab.textContent = shortTeam(team);

      const hit = el("rect", { class: "hit", x: L, y: y(Math.max(...pts.map((p) => p.strength))) - 8,
        width: W - L - R, height: 16 }, svg);
      interactive(hit, `<div class="t-name"><span class="chip" style="background:${colors[team]}"></span>${team}</div>`
        + tipRow("R1", fmt(pts[0].strength, 2))
        + tipRow(`R${last.round}`, fmt(last.strength, 2))
        + tipRow("Change", fmt(last.strength - pts[0].strength, 2)));
    });
  }

  devRedraw = draw;
  draw();

  const lg = document.getElementById("dev-legend");
  lg.innerHTML = `<li><span class="swatch line" style="background:${css("--series-1")}"></span>selected team (89% band)</li>`
    + `<li><span class="swatch line" style="background:${css("--axis")}"></span>other constructors</li>`;
}

/* =================== TEAMS =================== */

function renderTeams(t, news) {
  const chips = document.getElementById("team-chips");
  t.teams.forEach((team, i) => {
    const b = document.createElement("button");
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(i === 0));
    b.innerHTML = `<span class="chip" style="background:${t.team_colors_local?.[team.team] || "#898781"}"></span>`;
    b.append(document.createTextNode(shortTeam(team.team)));
    b.addEventListener("click", () => selectTeam(team.team));
    chips.appendChild(b);
  });

  const tb = document.querySelector("#standings-table tbody");
  t.teams.forEach((team) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${team.championship_position}</td><td class="name">${team.team}</td>`
      + `<td class="name">${team.power_unit || "—"}</td><td>${team.points}</td>`
      + `<td>${team.wins}</td><td>${team.podiums}</td><td>${team.dnf}</td>`
      + `<td>${team.pace_s === null ? "—" : fmt(team.pace_s, 3)}</td>`
      + `<td>${team.strength === null ? "—" : fmt(team.strength, 2)}</td>`;
    tb.appendChild(tr);
  });

  window.__teams = t;
  window.__news = news;
  selectTeam(t.teams[0].team);
}

function selectTeam(name) {
  state.selectedTeam = name;
  const t = window.__teams;
  const team = t.teams.find((x) => x.team === name);
  document.querySelectorAll("#team-chips button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.textContent.trim() === shortTeam(name))));

  const host = document.getElementById("team-detail");
  host.innerHTML = "";

  const head = document.createElement("div");
  head.className = "td-head";
  head.innerHTML = `<span class="chip" style="width:14px;height:14px;border-radius:4px;margin-top:7px;background:${t.team_colors_local?.[name] || "#898781"}"></span>`
    + `<div><div class="td-name">${team.team}</div></div>`;
  host.appendChild(head);

  const sub = document.createElement("div");
  sub.className = "td-sub";
  sub.textContent = [
    team.power_unit ? `${team.power_unit} power` + (team.works ? " (works)" : " (customer)") : null,
    team.first_season ? `in F1 since ${team.first_season}` : null,
    `P${team.championship_position} in the constructors' championship`,
  ].filter(Boolean).join(" · ");
  host.appendChild(sub);

  if (team.entry_note) {
    const n = document.createElement("div");
    n.className = "td-note";
    n.textContent = team.entry_note;
    host.appendChild(n);
  }

  const stats = document.createElement("div");
  stats.className = "td-stats";
  const devTxt = team.development === null ? "—" : fmt(team.development, 2);
  const paceDeltaTxt = team.pace_delta_s === null ? "—" : fmt(team.pace_delta_s, 2);
  [
    ["Points", team.points, ""],
    ["Wins", team.wins, `${team.podiums} podiums`],
    ["Best finish", team.best_finish ? "P" + team.best_finish : "—", ""],
    ["Retirements", team.dnf, `from ${team.starts} starts`],
    ["Avg. grid", team.avg_grid ?? "—", ""],
    ["Corrected pace", team.pace_s === null ? "—" : fmt(team.pace_s, 2) + " s", "vs field mean"],
    ["Car strength", team.strength === null ? "—" : fmt(team.strength, 2), "log-odds"],
    ["Development", devTxt, `pace trend ${paceDeltaTxt} s`],
  ].forEach(([k, v, s2]) => {
    const d = document.createElement("div");
    d.className = "td-stat";
    d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`
      + (s2 ? `<div class="k" style="margin-top:3px">${s2}</div>` : "");
    stats.appendChild(d);
  });
  host.appendChild(stats);

  const cols = document.createElement("div");
  cols.className = "td-cols";

  const dcol = document.createElement("div");
  dcol.innerHTML = `<h4 class="td-section-h">Drivers</h4>`;
  team.drivers.forEach((d) => {
    const row = document.createElement("div");
    row.className = "driver-row";
    const bits = [`${d.points} pts`, `${d.wins}W`, `${d.podiums}P`];
    if (d.pace_s !== null) bits.push(`${fmt(d.pace_s, 2)}s`);
    if (d.skill !== null) bits.push(`skill ${fmt(d.skill, 2)}`);
    row.innerHTML = `<span class="dr-code">${d.code}</span>`
      + `<span class="dr-name">${d.name}</span>`
      + `<span class="dr-num">${bits.join(" · ")}</span>`;
    dcol.appendChild(row);
  });
  cols.appendChild(dcol);

  const ncol = document.createElement("div");
  ncol.innerHTML = `<h4 class="td-section-h">Recent headlines mentioning this team</h4>`;
  const items = (window.__news?.items || []).filter((i) => i.teams.includes(name)).slice(0, 6);
  if (!items.length) {
    const p = document.createElement("div");
    p.className = "news-empty";
    p.textContent = "No headlines in the current feed mention this team.";
    ncol.appendChild(p);
  } else {
    const ul = document.createElement("ul");
    ul.className = "news-list";
    items.forEach((i) => ul.appendChild(newsItem(i, true)));
    ncol.appendChild(ul);
  }
  cols.appendChild(ncol);
  host.appendChild(cols);

  if (devRedraw) devRedraw();
}

/* =================== NEWS =================== */

function relTime(iso) {
  const then = new Date(iso).getTime();
  const mins = Math.max(0, Math.round((Date.now() - then) / 60000));
  if (mins < 60) return `${mins}m ago`;
  const h = Math.round(mins / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return d === 1 ? "yesterday" : `${d}d ago`;
}

/* Builds one headline. Every string from the feed goes in via textContent — this is
   third-party content and is never treated as markup. */
function newsItem(item, compact = false) {
  const li = document.createElement("li");
  const a = document.createElement("a");
  a.href = item.link;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  a.textContent = item.title;
  if (compact) a.style.fontSize = "13.5px";
  li.appendChild(a);

  const meta = document.createElement("div");
  meta.className = "news-meta";
  const src = document.createElement("span");
  src.className = "src";
  src.textContent = item.source;
  const when = document.createElement("span");
  when.textContent = relTime(item.published);
  meta.append(src, when);
  if (!compact && item.teams.length) {
    const tg = document.createElement("span");
    tg.textContent = item.teams.map(shortTeam).join(", ");
    meta.appendChild(tg);
  }
  li.appendChild(meta);

  if (!compact && item.summary) {
    const p = document.createElement("div");
    p.className = "news-summary";
    p.textContent = item.summary;
    li.appendChild(p);
  }
  return li;
}

function renderNews(news, teamNames) {
  const filter = document.getElementById("news-filter");
  const mk = (label, value) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.setAttribute("aria-pressed", String(state.newsFilter === value));
    b.addEventListener("click", () => {
      state.newsFilter = state.newsFilter === value ? null : value;
      drawList();
      [...filter.children].forEach((c) =>
        c.setAttribute("aria-pressed", String(c.dataset.v === String(state.newsFilter))));
    });
    b.dataset.v = String(value);
    return b;
  };
  filter.appendChild(mk("All", null));
  teamNames.forEach((t) => filter.appendChild(mk(shortTeam(t), t)));

  const list = document.getElementById("news-list");
  function drawList() {
    list.innerHTML = "";
    const items = news.items.filter((i) =>
      !state.newsFilter || i.teams.includes(state.newsFilter));
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "news-empty";
      li.textContent = "No headlines in the current feed match that team.";
      list.appendChild(li);
      return;
    }
    items.slice(0, 24).forEach((i) => list.appendChild(newsItem(i)));
  }
  drawList();

  const src = document.getElementById("news-sources");
  src.textContent = `Sources: ${news.sources.join(", ")}. `
    + `${news.items.length} headlines fetched ${relTime(news.generated_utc)}. `
    + `Headlines and excerpts belong to their publishers; follow a link to read the article.`;
}

/* =================== CALIBRATION =================== */

const METRICS = ["rps", "ll_win", "ll_podium", "ll_points", "spearman"];
const LOWER_IS_BETTER = { rps: 1, ll_win: 1, ll_podium: 1, ll_points: 1, spearman: 0 };

function renderCalibration(s) {
  const c = s.calibration;
  const verdict = document.getElementById("verdict");
  const note = document.getElementById("calib-note");

  if (!c) {
    verdict.innerHTML = `<span class="v-head">Not yet validated</span>`
      + `<b>This forecast has never been scored out of sample.</b> Until the walk-forward `
      + `backtest is run, there is no evidence that these probabilities are any better than `
      + `guessing from the grid.`;
    return;
  }

  const rows = [...c.summary].sort((a, b) => a.rps - b.rps);
  const best = {};
  METRICS.forEach((m) => {
    best[m] = rows.reduce((acc, r) =>
      (LOWER_IS_BETTER[m] ? r[m] < acc[m] : r[m] > acc[m]) ? r : acc, rows[0])[m];
  });

  const tb = document.querySelector("#calib-table tbody");
  tb.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    if (r.model === c.best_model) tr.className = "is-best";
    const cells = METRICS.map((m) =>
      `<td class="${r[m] === best[m] ? "win" : ""}">${r[m].toFixed(4)}</td>`).join("");
    tr.innerHTML = `<td class="name">${r.model}</td>${cells}`;
    tb.appendChild(tr);
  });

  // Whichever variant actually won, rather than a hard-coded name — the production
  // likelihood is chosen by this table, so the copy must follow it.
  const modelRow = rows.find((r) => r.model === c.best_model && r.model.startsWith("model:"))
                || rows.find((r) => r.model.startsWith("model:"));
  const gridRow = rows.find((r) => r.model === "baseline: grid");
  const gap = modelRow && gridRow ? modelRow.rps - gridRow.rps : null;

  const sig = c.significance || {};
  if (c.beats_grid_baseline) {
    // "Best on the mean" and "better" are different claims. With six races the paired
    // margin is nowhere near significant, and the banner has to say so — leading with
    // the win and burying the t-statistic would be the dishonest version of this page.
    verdict.className = "verdict" + (sig.significant ? " pass" : "");
    verdict.innerHTML = `<span class="v-head">Best on the mean — not yet proven</span>`
      + `<b>Lowest RPS of anything tested</b> across ${c.n_eval_races} unseen races, `
      + `including qualifying position `
      + `(${modelRow ? modelRow.rps.toFixed(3) : "—"} vs ${c.grid_baseline_rps.toFixed(3)}). `
      + (sig.t_stat !== undefined
          ? `But it wins only ${sig.races_won} of ${c.n_eval_races} races and the paired `
            + `margin is <b>t = ${sig.t_stat}</b> — indistinguishable from noise at this `
            + `sample size. Treat it as promising, not established.`
          : "");
  } else {
    verdict.className = "verdict";
    verdict.innerHTML = `<span class="v-head">Read the forecast with this in mind</span>`
      + `<b>It does not beat the grid baseline.</b> Qualifying position alone scores a better `
      + `RPS (${c.grid_baseline_rps.toFixed(3)}`
      + (gap !== null ? ` vs ${modelRow.rps.toFixed(3)}` : "") + `) over ${c.n_eval_races} `
      + `out-of-sample races. Treat the probabilities below with that in mind.`;
  }

  note.innerHTML = `Widest margin is the midfield: points log-loss `
    + `${modelRow ? modelRow.ll_points.toFixed(3) : "—"} against `
    + `${gridRow ? gridRow.ll_points.toFixed(3) : "—"} for the grid. `
    + `<b>forward</b> reads the order the obvious way round and loses; <b>contaminated</b> `
    + `tried to fix that and did not. Both are kept as the evidence for the one that ships. `
    + `<a href="method.html#calibration">More →</a>`;
}

/* =================== LAYER 0: CORRECTED PACE =================== */

function renderPace(data) {
  const rows = data.pace;
  const host = document.getElementById("pace-chart");
  const ROW = 27, BAR = 14, PAD_T = 26, PAD_B = 34, GUTTER = 132, PAD_R = 62;
  const W = 860, H = PAD_T + rows.length * ROW + PAD_B;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Fuel- and tyre-corrected race pace by driver, seconds relative to field mean" }, host);

  const lo = Math.min(...rows.map((r) => r.pace_s - r.se_s));
  const hi = Math.max(...rows.map((r) => r.pace_s + r.se_s));
  const span = Math.max(Math.abs(lo), Math.abs(hi)) * 1.12;
  const x = (v) => GUTTER + ((v + span) / (2 * span)) * (W - GUTTER - PAD_R);
  const x0 = x(0);

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
    const y = PAD_T + i * ROW, yb = y + (ROW - BAR) / 2;
    el("rect", { x: 0, y: yb + 3, width: 8, height: 8, rx: 2,
      fill: data.team_colors[r.Team] || css("--ink-muted") }, svg);
    const nm = el("text", { class: "name-label" + (i < 3 ? " lead" : ""), x: 14, y: yb + BAR - 3 }, svg);
    nm.textContent = r.Driver;
    const tm = el("text", { x: 58, y: yb + BAR - 3 }, svg);
    tm.textContent = shortTeam(r.Team);

    el("path", { d: barPath(x0, x(r.pace_s), yb, BAR), fill: css("--series-1") }, svg);

    const eL = x(r.pace_s - r.se_s), eR = x(r.pace_s + r.se_s), yc = yb + BAR / 2;
    el("line", { class: "err", x1: eL, x2: eR, y1: yc, y2: yc }, svg);
    el("line", { class: "err", x1: eL, x2: eL, y1: yc - 3.5, y2: yc + 3.5 }, svg);
    el("line", { class: "err", x1: eR, x2: eR, y1: yc - 3.5, y2: yc + 3.5 }, svg);

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

  const tb = document.querySelector("#pace-table tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.rank}</td><td class="name">${r.Driver}</td><td class="name">${r.Team}</td>`
      + `<td>${fmt(r.pace_s)}</td><td>${fmtAbs(r.se_s)}</td><td>${fmtAbs(r.sd_s)}</td>`
      + `<td>${r.races}</td><td>${r.laps}</td>`;
    tb.appendChild(tr);
  });
}

/* =================== TYRE DEGRADATION =================== */

function renderDeg(data) {
  const comps = data.degradation.filter((d) => ["SOFT", "MEDIUM", "HARD"].includes(d.compound));
  const host = document.getElementById("deg-chart");
  const W = 560, H = 300, L = 46, R = 74, T = 18, B = 42;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Modelled cumulative time loss versus tyre age, by compound" }, host);

  const MAXAGE = 30;
  const maxLoss = Math.max(0.6, ...comps.map((c) => c.deg_s_per_lap * MAXAGE)) * 1.1;
  const x = (a) => L + (a / MAXAGE) * (W - L - R);
  const y = (v) => H - B - (v / maxLoss) * (H - T - B);

  // Compound hardness is an ordered category, so darkness carries the order.
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

/* =================== PACE BY ROUND =================== */

function renderForm(data) {
  const order = data.pace.map((p) => p.Driver);
  const rounds = data.rounds_analysed;
  const idx = new Map();
  data.pace_by_round.forEach((r) => idx.set(`${r.Driver}|${r.round}`, r));
  const eventOf = new Map(data.pace_by_round.map((r) => [r.round, r.event]));

  const host = document.getElementById("form-chart");
  const CELL = 26, GAPC = 2, ROW = 22, GUTTER = 58, T = 30, B = 12;
  const W = GUTTER + rounds.length * (CELL + GAPC) + 8;
  const H = T + order.length * ROW + B;
  const svg = el("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`,
    style: `min-width:${W}px`, role: "img",
    "aria-label": "Corrected pace by driver and round, relative to the field mean in each race" }, host);

  const cap = Math.max(0.4, ...data.pace_by_round.map((r) => Math.abs(r.pace_s))) * 0.92;
  const cool = css("--div-cool"), warm = css("--div-warm"), mid = css("--div-mid");
  const color = (v) => {
    const t = Math.min(1, Math.abs(v) / cap);
    return v < 0 ? mixHex(mid, cool, t) : mixHex(mid, warm, t);
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
        // midpoint is deliberately faint, so absence is marked as a state, not a value.
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

  document.getElementById("form-legend").innerHTML =
    `<li><span class="swatch" style="background:${mixHex(mid, cool, 1)}"></span>faster than field mean</li>`
    + `<li><span class="swatch" style="background:${mid}"></span>at field mean</li>`
    + `<li><span class="swatch" style="background:${mixHex(mid, warm, 1)}"></span>slower than field mean</li>`
    + `<li><span class="swatch" style="background:linear-gradient(45deg,transparent 44%,${css("--axis")} 44%,${css("--axis")} 56%,transparent 56%)"></span>no valid race laps</li>`;

  document.querySelector("#form-table thead tr").innerHTML =
    "<th>Driver</th>" + rounds.map((r) => `<th>R${r}</th>`).join("");
  const tb = document.querySelector("#form-table tbody");
  order.forEach((drv) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="name">${drv}</td>`
      + rounds.map((rd) => {
        const rec = idx.get(`${drv}|${rd}`);
        return `<td>${rec ? fmt(rec.pace_s, 2) : "—"}</td>`;
      }).join("");
    tb.appendChild(tr);
  });
}

/* =================== HEADER / TILES =================== */

function renderHeader(data, s) {
  const ne = data.next_event;

  // Countdown lives in the eyebrow now — the hero's big number is the forecast, which
  // is the thing worth leading with.
  const [yy, mm, dd] = ne.date.split("-").map(Number);
  const raceDay = Date.UTC(yy, mm - 1, dd);
  const now = new Date();
  const today = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  const days = Math.max(0, Math.round((raceDay - today) / 86400000));
  const when = days === 0 ? "Today" : days === 1 ? "Tomorrow" : `In ${days} days`;

  document.getElementById("hero-eyebrow").textContent =
    `Round ${ne.round} of ${ne.total_rounds} · ${when}`;
  document.getElementById("race-name").textContent = ne.name;

  const meta = document.getElementById("race-meta");
  const base = `${ne.location}, ${ne.country} · ${ne.date}`
    + (ne.format.includes("sprint") ? " · sprint weekend" : "");
  meta.dataset.base = base;      // hero.js appends the pole time once the trace loads
  meta.textContent = base;

  const fav = [...s.forecast].sort((a, b) => b.p_win - a.p_win)[0];
  document.getElementById("hero-stat").textContent = pct(fav.p_win);
  document.getElementById("hero-stat-k").textContent = `${fav.driver} to win`;

  const built = "built " + data.generated_utc.replace("T", " ");
  document.getElementById("stamp").textContent = built;
  document.getElementById("foot-stamp").textContent = built;

  const t = data.totals;
  const tiles = [
    ["Pole", s.forecast.find((f) => f.grid === 1)?.driver ?? "—",
      `${fav.driver} favourite at ${pct(fav.p_win)}`],
    ["Green-flag laps modelled", t.laps_modelled.toLocaleString(),
      "fuel, tyre and traffic corrected"],
    ["Explained by the car", pct(s.diagnostics.constructor_share, 0),
      "rest is the driver"],
    ["Cost of dirty air", fmtAbs(t.mean_dirty_air_cost_s, 2) + " s",
      "at zero gap, mean across races"],
  ];
  const host = document.getElementById("tiles");
  tiles.forEach(([label, value, sub]) => {
    const d = document.createElement("div");
    d.className = "tile";
    d.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>`
      + `<div class="sub">${sub}</div>`;
    host.appendChild(d);
  });

  document.getElementById("rho-note").textContent =
    `Spearman \u03c1 = ${s.diagnostics.layer0_spearman}.`;
  document.getElementById("conv-note").textContent =
    `Sampler converged: worst R-hat ${s.diagnostics.worst_rhat} over `
    + `${s.diagnostics.n_races} races.`;
}

/* =================== BOOT =================== */

const load = (p) => fetch(p).then((r) => {
  if (!r.ok) throw new Error(`${p} → HTTP ${r.status}`);
  return r.json();
});

Promise.all([
  load("data/pace_2026.json"),
  load("data/strength_2026.json"),
  load("data/teams_2026.json"),
  load("data/news.json").catch(() => ({ items: [], sources: [], generated_utc: new Date().toISOString() })),
]).then(([pace, strength, teams, news]) => {
  const colors = pace.team_colors;
  teams.team_colors_local = colors;

  renderHeader(pace, strength);
  renderCalibration(strength);
  renderForecast(strength, colors);
  renderMatrix(strength, colors);
  renderPace(pace);
  renderDeg(pace);
  renderForm(pace);
  renderDev(strength, colors);
  renderSplit(strength, colors, new Set(strength.position_matrix.drivers));
  renderTeams(teams, news);
  renderNews(news, teams.teams.map((t) => t.team));
}).catch((e) => {
  document.getElementById("tiles").innerHTML =
    `<div class="tile"><div class="label">Data</div><div class="value">—</div>`
    + `<div class="sub">Could not load a payload (${e.message}). Run `
    + `<code>make all</code> and <code>scripts/export_news.py</code>.</div></div>`;
});
