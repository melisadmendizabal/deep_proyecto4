/* Fight Diff - frontend (B13). Habla con la API de app/api.py (/health, /player, /analyze). */
"use strict";

const C = {
  ink: "#f0e6d2", muted: "#a09b8c", gold: "#c8aa6e", surface: "#0a1428",
  blue: "#3a8ee6", red: "#e8505b", grid: "rgba(160,155,140,0.16)",
};
const GROUPS = ["economía", "combate", "objetivos", "visión"];
const S = { res: null, side: null, persp: "azul", chart: null, ddragon: null, active: null };

// ------------------------------------------------------------------ utilidades
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => `${Math.round(100 * p)} %`;
const mmss = (m) => `${String(m).padStart(2, "0")}:00`;
const show = (el, on = true) => el.classList.toggle("hidden", !on);

async function api(path, params) {
  const url = `${path}?${new URLSearchParams(params)}`;
  let r;
  try { r = await fetch(url); } catch { throw new Error("No hay conexión con el servidor de la app."); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `Error ${r.status}`);
  return body;
}

function message(html, kind = "error") {
  $("#search-msg").innerHTML = html ? `<div class="msg ${kind}">${html}</div>` : "";
}
function loading(text) {
  $("#search-msg").innerHTML = `<div class="loading"><span class="spinner"></span>${esc(text)}</div>`;
}
function busy(form, on) { form.querySelectorAll("button, input, select").forEach((el) => (el.disabled = on)); }

// ------------------------------------------------------------------ inicio
async function init() {
  try {
    const h = await api("/health", {});
    const pill = $("#status");
    pill.innerHTML = `<span class="dot ${h.api_riot ? "ok" : "off"}"></span><span>modelo ${esc(h.modelo)} · ${h.api_riot ? "API de Riot conectada" : "sin API de Riot (solo caché)"}</span>`;
    if (h.partidas_demo?.length) {
      const row = $("#demo-row");
      h.partidas_demo.forEach((mid) => {
        const b = document.createElement("button");
        b.className = "chip"; b.type = "button"; b.textContent = mid;
        b.onclick = () => analyze(mid, null);
        row.appendChild(b);
      });
      show(row);
    }
  } catch (e) {
    $("#status").innerHTML = `<span class="dot off"></span><span>servidor no disponible</span>`;
  }
  fetch("https://ddragon.leagueoflegends.com/api/versions.json").then((r) => r.json())
    .then((v) => (S.ddragon = v[0])).catch(() => {});
  // enlaces directos: /?match=NA1_123&lado=azul  ·  /?riot=Nombre%23TAG&plataforma=na1
  const q = new URLSearchParams(location.search);
  if (q.get("riot")) {
    $("#riot-id").value = q.get("riot");
    if (q.get("plataforma")) $("#platform").value = q.get("plataforma");
    $("#form-riot").requestSubmit();
  } else if (q.get("match")) {
    const lado = ["azul", "rojo"].includes(q.get("lado")) ? q.get("lado") : null;
    analyze(q.get("match"), lado);
  }
}

// pestañas
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", t === tab));
    show($("#form-riot"), tab.id === "tab-riot");
    show($("#form-match"), tab.id === "tab-match");
  });
});

// ------------------------------------------------------------------ jugador
$("#form-riot").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.currentTarget;
  const riot = $("#riot-id").value.trim();
  if (!riot.includes("#")) { message("Escribe el Riot ID completo, con el tag: <b>Nombre#TAG</b>."); return; }
  busy(form, true); loading("Buscando partidas ranked recientes…");
  try {
    const r = await api("/player", { riot_id: riot, platform: $("#platform").value, count: $("#count").value });
    message("");
    renderPlayer(r);
  } catch (e) { message(esc(e.message)); }
  finally { busy(form, false); }
});

function champIcon(name) {
  if (S.ddragon && name) {
    return `<img class="champ" alt="${esc(name)}" src="https://ddragon.leagueoflegends.com/cdn/${esc(S.ddragon)}/img/champion/${esc(name)}.png"
      onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'champ',textContent:'${esc(name).slice(0, 2)}'}))">`;
  }
  return `<div class="champ">${esc((name || "?").slice(0, 2))}</div>`;
}

function renderPlayer(r) {
  $("#player-t").textContent = `Partidas recientes · ${r.riot_id}`;
  const box = $("#matches");
  box.innerHTML = "";
  if (!r.partidas.length) {
    box.innerHTML = `<p class="empty">No hay partidas ranked Solo/Duo recientes para este jugador.</p>`;
  }
  r.partidas.forEach((p) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `match-card ${p.gano ? "win" : "loss"}`;
    b.disabled = !p.analizable;
    b.title = p.analizable ? "Analizar esta partida" : "Partida de menos de 15 min: no se analiza";
    b.innerHTML = `${champIcon(p.campeon)}
      <div><div class="line1">${esc(p.campeon)} <span class="meta">${esc(p.rol || "")}</span></div>
      <div class="line2"><span class="result ${p.gano ? "win" : "loss"}">${p.gano ? "VICTORIA" : "DERROTA"}</span> · ${esc(p.kda)} · ${p.duracion_min} min</div>
      <div class="line3">lado ${esc(p.lado)} · ${esc(p.inicio_utc)} UTC${p.analizada ? " · ya analizada" : ""}</div></div>`;
    b.onclick = () => analyze(p.match_id, p.lado);
    box.appendChild(b);
  });
  show($("#player-panel"));
}

// ------------------------------------------------------------------ análisis
$("#form-match").addEventListener("submit", (ev) => {
  ev.preventDefault();
  analyze($("#match-id").value.trim(), null);
});

async function analyze(mid, side) {
  loading(`Analizando ${mid}… (descargar la partida de Riot tarda ~1 s)`);
  try {
    const res = await api("/analyze", { match_id: mid });
    message("");
    S.res = res; S.side = side; S.persp = side ? "mio" : "azul"; S.active = null;
    render();
    $("#analysis").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { message(esc(e.message)); }
}

function flipped() { return S.persp === "mio" && S.side === "rojo"; }
function view(p) { return flipped() ? 1 - p : p; }
function perspLabel() { return S.persp === "mio" && S.side ? `tu equipo (${S.side})` : "el equipo azul"; }
function relation(team) { return S.side ? (team === S.side ? "a tu favor" : "en tu contra") : ""; }

function render() {
  const r = S.res;
  const all = [...r.momentos, ...r.candidatos];
  $("#an-title").textContent = r.match_id;
  $("#an-winner").innerHTML = `<span class="badge ${r.gana}">Ganó ${r.gana}</span>`;
  $("#an-side").innerHTML = S.side ? `<span class="badge neutral">Tu equipo: ${esc(S.side)}</span>` : "";
  const fuente = { riot: "descargada de Riot", raw_cache: "datos guardados", memoria: "caché", disco: "caché", demo: "partida de ejemplo" }[r.fuente] || r.fuente;
  $("#an-meta").textContent = `${r.duracion_min} min · parche ${r.parche} · modelo ${r.modelo.nombre} · ${fuente}`;
  $("#an-warnings").innerHTML = (r.avisos || []).map((w) => `<div class="msg">${esc(w)}</div>`).join("");

  const persp = $("#persp");
  show(persp, !!S.side);
  persp.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.p === S.persp));

  // fichas
  $("#t-mom").textContent = r.momentos.length;
  $("#t-mom-s").textContent = r.candidatos.length ? `+ ${r.candidatos.length} posibles` : "sin posibles extra";
  const rv = r.revision;
  $("#t-full").textContent = rv ? `${Math.round(rv.completa_min)} min` : "–";
  $("#t-tool").textContent = rv ? `${Math.round(rv.con_clips_min)} min` : "–";
  $("#t-tool-s").textContent = rv ? `${rv.ahorro_pct} % menos (estimado)` : "";
  const pf = r.curva[r.curva.length - 1].p_azul;
  $("#t-final").textContent = pct(view(pf));
  $("#t-final-s").textContent = `prob. de ${perspLabel()} al final`;

  $("#chart-sub").textContent = `· de que gane ${perspLabel()}, minuto a minuto`;
  drawChart();
  renderMoments($("#moments"), r.momentos, false);
  renderMoments($("#candidates"), r.candidatos, true);
  show($("#cand-box"), r.candidatos.length > 0);
  if (!r.momentos.length) {
    $("#moments").innerHTML = `<p class="empty">El modelo no encontró cambios grandes de probabilidad (≥ ${Math.round(100 * r.modelo.theta)} pts):
      la partida se decidió de forma gradual.${r.candidatos.length ? " Revisa los posibles momentos de abajo." : ""}</p>`;
  }
  renderTable(all);
  ["#analysis", "#chart-panel", "#moments-panel"].forEach((s) => show($(s)));
}

$("#persp").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b || !S.res) return;
  S.persp = b.dataset.p;
  render();
});

// ------------------------------------------------------------------ momentos
function eventsLine(evs) {
  const part = (team) => {
    const d = evs[team];
    const items = [];
    if (d.kills) items.push(`${d.kills} kill${d.kills > 1 ? "s" : ""}`);
    const count = (arr) => Object.entries(arr.reduce((a, x) => ((a[x] = (a[x] || 0) + 1), a), {}))
      .map(([k, n]) => (n > 1 ? `${n}× ${k}` : k));
    items.push(...count(d.objetivos), ...count(d.estructuras));
    return items.length ? `<b class="${team}">${team === "azul" ? "Azul" : "Rojo"}:</b> ${esc(items.join(", "))}` : "";
  };
  const txt = [part("azul"), part("rojo")].filter(Boolean).join(" &nbsp;·&nbsp; ");
  return txt || "sin kills ni objetivos en esa ventana (cambio por oro/experiencia)";
}

function renderMoments(box, list, cand) {
  box.innerHTML = "";
  list.forEach((m, i) => {
    const el = document.createElement("article");
    el.className = `moment ${m.favorece}${cand ? " cand" : ""}`;
    el.tabIndex = 0;
    el.dataset.minute = m.minuto;
    const share = m.explicacion.aporte;
    const why = GROUPS.map((g) => `<span>${g}</span><span class="bar"><i style="width:${Math.round(100 * (share[g] || 0))}%"></i></span><span>${Math.round(100 * (share[g] || 0))} %</span>`).join("");
    const shape = m.favorece === "azul" ? "▲" : "▼";
    const rel = relation(m.favorece);
    el.innerHTML = `
      <div class="num"><span>${cand ? "?" : i + 1}</span></div>
      <div>
        <div class="when">Min ${m.desde_min}–${m.minuto}</div>
        <div class="txt">${esc(m.texto.replace(/^min \d+–\d+: /, "").replace(/ \([^()]*\)$/, ""))}</div>
        <div class="evs">${eventsLine(m.eventos)}</div>
        <div class="why" aria-label="Qué explica el cambio">${why}</div>
        <div class="replay">En la repetición: de <code>${mmss(m.desde_min)}</code> a <code>${mmss(m.minuto)}</code></div>
      </div>
      <div class="side">
        <div class="delta ${m.favorece}">${shape} ${Math.abs(Math.round(m.delta_pts))} pts</div>
        <div class="badge ${m.favorece}">a favor de ${m.favorece}</div>
        ${rel ? `<div class="conf">${rel}</div>` : ""}
        <div class="conf">confianza ${esc(m.confianza)} · ${m.semillas_de_acuerdo}/3 modelos</div>
      </div>`;
    const act = () => focusMoment(m.minuto, true);
    el.addEventListener("click", act);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } });
    box.appendChild(el);
  });
}

function focusMoment(minute, fromList) {
  S.active = minute;
  document.querySelectorAll(".moment").forEach((el) => el.classList.toggle("active", +el.dataset.minute === minute));
  const ch = S.chart;
  if (ch) {
    const idx = S.res.curva.findIndex((c) => c.minuto === minute);
    const meta = ch.getDatasetMeta(2);
    const pt = meta.data[idx];
    if (pt) {
      ch.setActiveElements([{ datasetIndex: 2, index: idx }]);
      ch.tooltip.setActiveElements([{ datasetIndex: 2, index: idx }], { x: pt.x, y: pt.y });
      ch.update();
    }
    if (fromList) $("#chart-panel").scrollIntoView({ behavior: "smooth", block: "center" });
  }
  if (!fromList) {
    const card = document.querySelector(`.moment[data-minute="${minute}"]`);
    card?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function renderTable(all) {
  const byMin = Object.fromEntries(all.map((m) => [m.minuto, m]));
  const rows = S.res.curva.map((c) => {
    const m = byMin[c.minuto];
    const tag = m ? `${m.candidato ? "posible" : "momento"} (${m.favorece})` : "";
    return `<tr><td>${c.minuto}</td><td>${pct(view(c.p_azul))}</td><td>${pct(Math.min(view(c.p_min), view(c.p_max)))}–${pct(Math.max(view(c.p_min), view(c.p_max)))}</td><td style="text-align:left">${tag}${c.baja_certeza ? " baja certeza" : ""}</td></tr>`;
  }).join("");
  $("#curve-table").innerHTML = `<thead><tr><th>Minuto</th><th>P(gana ${esc(perspLabel())})</th><th>Rango 3 modelos</th><th style="text-align:left">Nota</th></tr></thead><tbody>${rows}</tbody>`;
}

// ------------------------------------------------------------------ gráfico
// interacción: siempre el punto de la CURVA (dataset 2) más cercano en x, aunque el puntero esté
// sobre un marcador; el tooltip agrega el momento de ese minuto si lo hay.
Chart.Interaction.modes.curveX = function (chart, e) {
  const pos = Chart.helpers.getRelativePosition(e, chart);
  const meta = chart.getDatasetMeta(2);
  let best = null;
  meta.data.forEach((el, index) => {
    const d = Math.abs(el.x - pos.x);
    if (!best || d < best.d) best = { d, element: el, datasetIndex: 2, index };
  });
  return best ? [best] : [];
};

let hatch = null;
function hatchPattern(ctx) {
  if (hatch) return hatch;
  const c = document.createElement("canvas");
  c.width = c.height = 8;
  const g = c.getContext("2d");
  g.strokeStyle = "rgba(160,155,140,0.35)"; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(0, 8); g.lineTo(8, 0); g.stroke();
  hatch = ctx.createPattern(c, "repeat");
  return hatch;
}

const zonesPlugin = {
  id: "zones",
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea: a, scales: { x, y } } = chart;
    if (!S.res) return;
    ctx.save();
    // baja certeza: minutos 0-4
    ctx.fillStyle = hatchPattern(ctx);
    ctx.fillRect(x.getPixelForValue(0), a.top, x.getPixelForValue(4.5) - x.getPixelForValue(0), a.bottom - a.top);
    // ventana de cada clip
    for (const m of [...S.res.momentos, ...S.res.candidatos]) {
      const col = m.candidato ? "rgba(160,155,140,0.08)" : (m.favorece === "azul" ? "rgba(58,142,230,0.13)" : "rgba(232,80,91,0.13)");
      ctx.fillStyle = S.active === m.minuto ? col.replace(/0\.\d+\)$/, "0.28)") : col;
      ctx.fillRect(x.getPixelForValue(m.desde_min), a.top, x.getPixelForValue(m.minuto) - x.getPixelForValue(m.desde_min), a.bottom - a.top);
    }
    // 50 %
    ctx.strokeStyle = "rgba(160,155,140,0.55)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(a.left, y.getPixelForValue(0.5)); ctx.lineTo(a.right, y.getPixelForValue(0.5)); ctx.stroke();
    ctx.restore();
  },
  afterDatasetsDraw(chart) {
    const { ctx, scales: { x, y } } = chart;
    if (!S.res) return;
    ctx.save();
    ctx.font = "700 12px Cinzel, Georgia, serif"; ctx.textAlign = "center"; ctx.fillStyle = C.ink;
    S.res.momentos.forEach((m, i) => {   // número de cada momento junto a su marcador
      const yy = y.getPixelForValue(view(m.p_despues));
      const up = view(m.p_despues) > view(m.p_antes);
      ctx.fillText(String(i + 1), x.getPixelForValue(m.minuto), up ? yy - 16 : yy + 24);
    });
    // guía vertical del hover
    const act = chart.tooltip?.getActiveElements?.() || [];
    if (act.length) {
      const px = act[0].element.x;
      ctx.strokeStyle = "rgba(200,170,110,0.6)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, chart.chartArea.top); ctx.lineTo(px, chart.chartArea.bottom); ctx.stroke();
    }
    ctx.restore();
  },
};

function drawChart() {
  const r = S.res;
  const pts = (key) => r.curva.map((c) => ({ x: c.minuto, y: view(c[key]) }));
  const lo = r.curva.map((c) => ({ x: c.minuto, y: Math.min(view(c.p_min), view(c.p_max)) }));
  const hi = r.curva.map((c) => ({ x: c.minuto, y: Math.max(view(c.p_min), view(c.p_max)) }));
  const mk = (list) => list.map((m) => ({ x: m.minuto, y: view(m.p_despues), m }));
  const momentPts = mk(r.momentos);
  const candPts = mk(r.candidatos);
  const byMin = Object.fromEntries([...r.momentos.map((m, i) => [m.minuto, { m, n: i + 1 }]), ...r.candidatos.map((m) => [m.minuto, { m, n: null }])]);
  const upFor = (m) => view(m.p_despues) > view(m.p_antes);

  if (S.chart) S.chart.destroy();
  S.chart = new Chart($("#chart"), {
    type: "line",
    data: {
      datasets: [
        { data: lo, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false },
        { data: hi, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: "-1", backgroundColor: "rgba(240,230,210,0.16)" },
        { data: pts("p_azul"), borderColor: C.ink, borderWidth: 2, pointRadius: 0, pointHoverRadius: 5,
          pointHoverBackgroundColor: C.ink, pointHoverBorderColor: C.surface, pointHoverBorderWidth: 2, tension: 0.25 },
        { type: "scatter", data: momentPts, pointStyle: "triangle", pointRadius: 9, pointHoverRadius: 11,
          pointRotation: (ctx) => (ctx.raw && !upFor(ctx.raw.m) ? 180 : 0),
          backgroundColor: (ctx) => (ctx.raw?.m.favorece === "rojo" ? C.red : C.blue),
          borderColor: C.surface, borderWidth: 2 },
        { type: "scatter", data: candPts, pointStyle: "circle", pointRadius: 6, pointHoverRadius: 8,
          backgroundColor: "rgba(0,0,0,0)", borderColor: C.muted, borderWidth: 2 },
      ],
    },
    options: {
      // sin animación: el gráfico sale idéntico en capturas/impresión y respeta "reducir movimiento"
      responsive: true, maintainAspectRatio: false, animation: false,
      parsing: false, normalized: true,
      interaction: { mode: "curveX", intersect: false },
      scales: {
        x: { type: "linear", min: 0, max: r.curva.length - 1,
             ticks: { stepSize: 5, color: C.muted }, grid: { color: C.grid, drawTicks: false }, border: { display: false },
             title: { display: true, text: "minuto de la partida", color: C.muted } },
        y: { min: 0, max: 1, ticks: { stepSize: 0.25, color: C.muted, callback: (v) => `${Math.round(v * 100)} %` },
             grid: { color: C.grid, drawTicks: false }, border: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(1,10,19,0.95)", borderColor: C.gold, borderWidth: 1, padding: 10,
          titleColor: C.gold, bodyColor: C.ink, titleFont: { family: "Cinzel", weight: "700" },
          filter: (item) => item.datasetIndex === 2,
          callbacks: {
            title: (items) => (items.length ? `Minuto ${items[0].raw.x}` : ""),
            label: (item) => {
              const c = r.curva[item.dataIndex];
              const a = Math.min(view(c.p_min), view(c.p_max)), b = Math.max(view(c.p_min), view(c.p_max));
              return [`P(gana ${perspLabel()}): ${pct(view(c.p_azul))}`, `rango 3 modelos: ${pct(a)}–${pct(b)}`]
                .concat(c.baja_certeza ? ["(inicio de partida: baja certeza)"] : []);
            },
            afterBody: (items) => {
              const hit = items.length && byMin[items[0].raw.x];
              if (!hit) return [];
              const m = hit.m;
              return ["", `${m.favorece === "azul" ? "▲" : "▼"} ${hit.n ? `Momento ${hit.n}` : "Posible momento"}: ${Math.abs(Math.round(m.delta_pts))} pts a favor de ${m.favorece}`,
                      `sobre todo por ${m.explicacion.grupo}`];
            },
          },
        },
      },
      onClick: (ev, els, chart) => {
        const near = chart.getElementsAtEventForMode(ev, "curveX", { intersect: false }, false)[0];
        if (!near) return;
        const minute = r.curva[near.index].minuto;
        if (byMin[minute]) focusMoment(minute, false);
      },
    },
    plugins: [zonesPlugin],
  });
}

init();
