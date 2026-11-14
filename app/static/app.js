/* Fight Diff - frontend (B13). Habla con la API de app/api.py (/health, /player, /analyze). */
"use strict";

const C = {
  ink: "#f0e6d2", muted: "#a09b8c", gold: "#c8aa6e", surface: "#0a1428",
  blue: "#3a8ee6", red: "#e8505b", grid: "rgba(160,155,140,0.16)",
};
const GROUPS = ["economía", "combate", "objetivos", "visión"];
// Opacidades de las zonas del gráfico (la leyenda de style.css usa las mismas).
const ZONE = { win: 0.26, winActive: 0.45, edge: 0.8, cand: 0.16, candActive: 0.3, band: 0.3, hatch: 0.6, half: 0.75 };
const ROLE = { TOP: "Top", JUNGLE: "Jungla", MIDDLE: "Mid", BOTTOM: "ADC", UTILITY: "Support" };
const PULSE_MS = 1600;   // parpadeo del punto al elegir una fila de la tabla
const ICON = {
  ok: `<svg class="ico" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M4.6 8.3l2.3 2.3 4.6-4.8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  pending: `<svg class="ico" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="2.6 2.2"/><circle cx="8" cy="8" r="1.6" fill="currentColor"/></svg>`,
  no: `<svg class="ico" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M3.3 12.7l9.4-9.4" stroke="currentColor" stroke-width="1.5"/></svg>`,
};
// team: de qué equipo se muestra la probabilidad ("azul"/"rojo"); side: el equipo del jugador (si se sabe)
const S = {
  res: null, side: null, team: "azul", model: null, models: [], chart: null, ddragon: null, active: null,
  player: null, selected: null, demos: [], demoSel: -1, pulse: null, pulseRaf: 0,
};

// ------------------------------------------------------------------ utilidades
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => `${Math.round(100 * p)} %`;
const mmss = (m) => `${String(m).padStart(2, "0")}:00`;
const show = (el, on = true) => el.classList.toggle("hidden", !on);
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

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

function setStatus(ok, text, title = "") {
  const el = $("#status");
  el.classList.toggle("off", !ok);
  el.title = title;
  // conectada: solo el texto; sin conexión: signo de exclamación dorado
  el.innerHTML = `${ok ? "" : `<span class="warn-ico" aria-hidden="true">!</span>`}<span>${esc(text)}</span>`;
}

// ------------------------------------------------------------------ inicio
async function init() {
  let apiOk = false;
  try {
    const h = await api("/health", {});
    apiOk = !!h.api_riot;
    if (apiOk) setStatus(true, "API de Riot conectada");
    else setStatus(false, "API de Riot sin conexión", "Solo funcionan las partidas guardadas (pestaña «Uso sin conexión»)");
    setupModels(h.modelos || [], h.modelo);
    setupDemos(h.demos || []);
  } catch (e) {
    setStatus(false, "Servidor no disponible");
  }
  fetch("https://ddragon.leagueoflegends.com/api/versions.json").then((r) => r.json())
    .then((v) => (S.ddragon = v[0])).catch(() => {});
  // enlaces directos: /?match=NA1_123&lado=azul&modelo=M2  ·  /?riot=Nombre%23TAG&plataforma=na1
  const q = new URLSearchParams(location.search);
  if (q.get("riot")) {
    $("#riot-id").value = q.get("riot");
    if (q.get("plataforma")) $("#platform").value = q.get("plataforma");
    $("#form-riot").requestSubmit();
  } else if (q.get("match")) {
    const lado = ["azul", "rojo"].includes(q.get("lado")) ? q.get("lado") : null;
    if (q.get("modelo") && S.models.some((m) => m.nombre === q.get("modelo"))) setModel(q.get("modelo"));
    analyze(q.get("match"), lado);
  } else if (!apiOk && S.demos.length) {
    selectTab("tab-demo");     // sin API de Riot solo sirven las partidas guardadas
  }
}

// ------------------------------------------------------------------ pestañas y ayudas
const TABS = { "tab-riot": "#form-riot", "tab-match": "#form-match", "tab-demo": "#form-demo" };
function selectTab(id) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.setAttribute("aria-selected", t.id === id);
    t.tabIndex = t.id === id ? 0 : -1;
  });
  Object.entries(TABS).forEach(([tid, sel]) => show($(sel), tid === id));
  show($("#player-panel"), id === "tab-riot" && !!S.player);   // la lista pertenece a la búsqueda por usuario
  closeHelps();
  closeDemoList();
  message("");
}
document.querySelectorAll(".tab").forEach((tab, i, all) => {
  tab.addEventListener("click", () => selectTab(tab.id));
  tab.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const next = all[(i + (e.key === "ArrowRight" ? 1 : all.length - 1)) % all.length];
    selectTab(next.id); next.focus();
  });
});

// las ayudas y la lista desplegable flotan sobre la página (position: fixed, colgadas de <body>: el
// clip-path de los paneles recorta incluso a los hijos fixed); se colocan bajo su ancla y se cierran al hacer scroll
document.querySelectorAll(".help, .dd-list").forEach((el) => document.body.appendChild(el));
function placeBelow(pop, anchor, matchWidth = false) {
  const r = anchor.getBoundingClientRect();
  if (matchWidth) pop.style.width = `${r.width}px`;
  pop.style.top = `${r.bottom + 6}px`;
  pop.style.left = `${Math.max(8, Math.min(r.left, innerWidth - pop.offsetWidth - 8))}px`;
}
const closePopups = (e) => {
  if (e?.target instanceof Element && e.target.closest(".help, .dd-list")) return;   // scroll dentro de la lista
  closeHelps(); closeDemoList();
};
addEventListener("scroll", closePopups, true);
addEventListener("resize", () => closePopups());

function closeHelps(except) {
  document.querySelectorAll(".info").forEach((b) => {
    if (b === except) return;
    b.setAttribute("aria-expanded", "false");
    show(document.getElementById(b.getAttribute("aria-controls")), false);
  });
}
document.querySelectorAll(".info").forEach((b) => {
  b.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = b.getAttribute("aria-expanded") !== "true";
    closeHelps(b);
    b.setAttribute("aria-expanded", String(open));
    const help = document.getElementById(b.getAttribute("aria-controls"));
    show(help, open);
    if (open) placeBelow(help, b.closest(".label-row"));
  });
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".help")) closeHelps();
  if (!e.target.closest("#demo-dd, .dd-list")) closeDemoList();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeHelps(); });

// ------------------------------------------------------------------ selector de modelo
function setupModels(models, main) {
  S.models = models;
  S.model = main;
  const sel = $("#model");
  sel.innerHTML = models.map((m) => `<option value="${esc(m.nombre)}">${esc(m.nombre)} · ${esc(m.etiqueta)}</option>`).join("");
  sel.value = main;
  show($("#model-pick"), models.length > 1);
}
function setModel(name) { S.model = name; $("#model").value = name; }

$("#model").addEventListener("change", (ev) => {
  S.model = ev.target.value;
  if (S.res) analyze(S.res.match_id, S.side, true);   // misma partida y perspectiva, otro modelo
});

function modelNote(r) {
  const info = S.models.find((m) => m.nombre === r.modelo.nombre);
  const main = S.models.find((m) => m.principal);
  if (!info || info.principal) return "";
  const auc = (x) => (x == null ? "" : `AUC en test ${x.toFixed(3)}`);
  return `Estás viendo <b>${esc(info.nombre)} · ${esc(info.etiqueta)}</b>: ${esc(info.descripcion)} (${auc(info.auc_test)}).
    El modelo principal es <b>${esc(main?.nombre || "M4")}</b>${main?.auc_test ? ` (${auc(main.auc_test)})` : ""}.
    Cada modelo usa su propio umbral de momento: aquí θ = ${Math.round(100 * r.modelo.theta)} pts.`;
}

// ------------------------------------------------------------------ cantidad de partidas (1-10, enteros)
const countIn = $("#count");
let countTimer = 0;
function countWarn(text) {
  $("#count-msg").textContent = text;
  clearTimeout(countTimer);
  if (text) countTimer = setTimeout(() => ($("#count-msg").textContent = ""), 3500);
}
countIn.addEventListener("keydown", (e) => {
  if (["e", "E", "+", "-", ".", ","].includes(e.key)) {
    e.preventDefault();
    countWarn(e.key === "-" ? "No se permiten negativos" : "Solo números enteros");
  }
});
countIn.addEventListener("paste", (e) => {
  if (!/^\d+$/.test((e.clipboardData?.getData("text") || "").trim())) { e.preventDefault(); countWarn("Solo números enteros"); }
});
countIn.addEventListener("input", () => {
  if (countIn.value === "") return;               // se permite borrar mientras escribe
  const n = Number(countIn.value);
  if (!Number.isInteger(n)) { countIn.value = String(Math.max(1, Math.min(10, Math.trunc(n) || 1))); countWarn("Solo números enteros"); }
  else if (n < 1) { countIn.value = "1"; countWarn("Mínimo 1 partida"); }
  else if (n > 10) { countIn.value = "10"; countWarn("Máximo 10 partidas"); }
});
function readCount() {
  const n = Number(countIn.value);
  if (countIn.value === "" || !Number.isInteger(n) || n < 1 || n > 10) {
    countWarn("Escribe un número del 1 al 10");
    countIn.focus();
    return null;
  }
  return n;
}

// ------------------------------------------------------------------ búsqueda por usuario
$("#form-riot").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.currentTarget;
  const riot = $("#riot-id").value.trim();
  if (!riot.includes("#")) { message("Escribe el Riot ID completo, con el tag: <b>Nombre#TAG</b>."); return; }
  const count = readCount();
  if (count == null) return;
  busy(form, true); loading("Buscando partidas ranked recientes…");
  try {
    const r = await api("/player", { riot_id: riot, platform: $("#platform").value, count });
    message("");
    S.selected = null;
    renderPlayer(r);
  } catch (e) { message(esc(e.message)); }
  finally { busy(form, false); }
});

function champIcon(name) {
  if (S.ddragon && name) {
    return `<img class="champ" alt="" src="https://ddragon.leagueoflegends.com/cdn/${esc(S.ddragon)}/img/champion/${esc(name)}.png"
      onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'champ',textContent:'${esc(name).slice(0, 2)}'}))">`;
  }
  return `<div class="champ" aria-hidden="true">${esc((name || "?").slice(0, 2))}</div>`;
}

function localDate(utc) {
  const d = new Date(String(utc).replace(" ", "T") + "Z");
  if (Number.isNaN(d.getTime())) return `${utc} UTC`;
  return d.toLocaleString("es", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function renderPlayer(r) {
  S.player = r;
  $("#player-t").textContent = `Partidas recientes · ${r.riot_id}`;
  const box = $("#matches");
  box.innerHTML = "";
  if (!r.partidas.length) {
    box.innerHTML = `<p class="empty">No hay partidas ranked Solo/Duo recientes para este jugador en ese servidor.</p>`;
  }
  r.partidas.forEach((p) => {
    const item = document.createElement("div");
    item.setAttribute("role", "listitem");
    const b = document.createElement("button");
    b.type = "button";
    const found = p.jugador_encontrado !== false;
    const sel = S.selected === p.match_id;
    b.className = `match-card ${found ? (p.gano ? "win" : "loss") : ""}${sel ? " selected" : ""}`;
    b.setAttribute("aria-pressed", String(sel));
    b.disabled = !p.analizable;
    b.title = p.analizable ? "Analizar esta partida" : "Partida de menos de 15 min: no se analiza";
    // support: puntaje de visión; los demás roles: farm (súbditos + jungla)
    const farm = !found ? ""
      : p.rol === "UTILITY" && p.vision != null ? `<span title="Puntaje de visión">Visión ${p.vision} · ${p.vision_min}/min</span>`
      : p.cs != null ? `<span title="Súbditos + monstruos de jungla">Farm ${p.cs} · ${p.cs_min}/min</span>` : "";
    const state = !p.analizable ? `<span class="st no">${ICON.no}Muy corta para analizar</span>`
      : p.analizada ? `<span class="st ok">${ICON.ok}Analizada</span>`
      : `<span class="st pending">${ICON.pending}Sin analizar</span>`;
    b.innerHTML = `${champIcon(p.campeon)}
      <div class="mc-main">
        ${found ? `<div class="result ${p.gano ? "win" : "loss"}">${p.gano ? "Victoria" : "Derrota"}</div>`
                : `<div class="result">Partida ranked</div>`}
        <div class="mc-champ">${found ? `${esc(p.campeon)} <span class="meta">· ${esc(ROLE[p.rol] || p.rol || "sin rol")}</span>`
                                      : `<span class="meta">no te encontramos en esta partida</span>`}</div>
        ${found ? `<div class="mc-stats"><span title="Asesinatos / muertes / asistencias">${esc(p.kda)}</span>${farm ? ` · ${farm}` : ""}</div>` : ""}
      </div>
      <div class="mc-side">
        <div>${p.duracion_min} min${found ? ` · lado <span class="${esc(p.lado)}">${esc(p.lado)}</span>` : ""}</div>
        <div>${esc(localDate(p.inicio_utc))}</div>
        <div>${state}</div>
      </div>`;
    b.onclick = () => {
      S.selected = p.match_id;
      document.querySelectorAll(".match-card").forEach((c) => {
        const on = c === b;
        c.classList.toggle("selected", on);
        c.setAttribute("aria-pressed", String(on));
      });
      analyze(p.match_id, p.lado);
    };
    item.appendChild(b);
    box.appendChild(item);
  });
  show($("#player-panel"), $('.tab[aria-selected="true"]').id === "tab-riot");
}

// ------------------------------------------------------------------ búsqueda por partida
$("#form-match").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const mid = $("#match-id").value.trim();
  if (!mid) { message("Escribe el ID de la partida (por ejemplo <b>NA1_5676698765</b>)."); return; }
  analyze(mid, null);
});

// ------------------------------------------------------------------ uso sin conexión (lista desplegable)
function demoOption(d, i) {
  const n = d.n_momentos;
  return `<span class="dd-main">Partida de ejemplo ${i + 1} <span class="dd-id">${esc(d.match_id)}</span></span>
    <span class="dd-sub">Ganó <b class="${esc(d.gana)}">${esc(d.gana)}</b> · ${d.duracion_min} min · ${n} momento${n === 1 ? "" : "s"} decisivo${n === 1 ? "" : "s"}</span>`;
}
function setupDemos(demos) {
  S.demos = demos;
  S.demoSel = demos.length ? 0 : -1;
  $("#demo-list").innerHTML = demos.map((d, i) =>
    `<li role="option" id="demo-opt-${i}" data-i="${i}" aria-selected="${i === S.demoSel}">${demoOption(d, i)}</li>`).join("");
  updateDemoBtn();
  if (!demos.length) $("#tab-demo").title = "No hay partidas guardadas";
}
function updateDemoBtn() {
  const d = S.demos[S.demoSel];
  $("#demo-dd-btn").innerHTML = d ? `<span class="dd-value">${demoOption(d, S.demoSel)}</span>` : `<span class="dd-value">No hay partidas guardadas</span>`;
  document.querySelectorAll("#demo-list li").forEach((li) => li.setAttribute("aria-selected", String(+li.dataset.i === S.demoSel)));
  $("#demo-list").setAttribute("aria-activedescendant", d ? `demo-opt-${S.demoSel}` : "");
}
function openDemoList() {
  if (!S.demos.length) return;
  show($("#demo-list"), true);
  placeBelow($("#demo-list"), $("#demo-dd-btn"), true);
  $("#demo-dd-btn").setAttribute("aria-expanded", "true");
  $("#demo-list").focus({ preventScroll: true });
}
function closeDemoList(focusBtn = false) {
  if ($("#demo-list").classList.contains("hidden")) return;
  show($("#demo-list"), false);
  $("#demo-dd-btn").setAttribute("aria-expanded", "false");
  if (focusBtn) $("#demo-dd-btn").focus();
}
$("#demo-dd-btn").addEventListener("click", () =>
  ($("#demo-list").classList.contains("hidden") ? openDemoList() : closeDemoList()));
$("#demo-dd-btn").addEventListener("keydown", (e) => {
  if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) { e.preventDefault(); openDemoList(); }
});
$("#demo-list").addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  S.demoSel = +li.dataset.i;
  updateDemoBtn();
  closeDemoList(true);
});
$("#demo-list").addEventListener("keydown", (e) => {
  const n = S.demos.length;
  if (e.key === "ArrowDown") S.demoSel = (S.demoSel + 1) % n;
  else if (e.key === "ArrowUp") S.demoSel = (S.demoSel + n - 1) % n;
  else if (e.key === "Enter" || e.key === " ") { closeDemoList(true); }
  else if (e.key === "Escape" || e.key === "Tab") { closeDemoList(e.key === "Escape"); return; }
  else return;
  e.preventDefault();
  updateDemoBtn();
  document.getElementById(`demo-opt-${S.demoSel}`)?.scrollIntoView({ block: "nearest" });
});
$("#form-demo").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const d = S.demos[S.demoSel];
  if (d) analyze(d.match_id, null);
});

// ------------------------------------------------------------------ análisis
async function analyze(mid, side, sameView = false) {
  const sel = $("#model");
  loading(`Analizando ${mid} con ${S.model || "el modelo principal"}… (descargar la partida de Riot tarda ~1 s)`);
  sel.disabled = true;
  try {
    const params = { match_id: mid };
    if (S.model) params.model = S.model;
    const res = await api("/analyze", params);
    message("");
    if (!sameView) { S.team = side || "azul"; S.active = null; }
    S.res = res; S.side = side;
    render();
    // la tarjeta de la lista pasa a "Analizada"
    const p = S.player?.partidas.find((x) => x.match_id === res.match_id);
    if (p && !p.analizada) { p.analizada = true; renderPlayer(S.player); }
    if (!sameView) $("#analysis").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  } catch (e) { message(esc(e.message)); }
  finally { sel.disabled = false; }
}

const other = (t) => (t === "azul" ? "rojo" : "azul");
function flipped() { return S.team === "rojo"; }
function view(p) { return flipped() ? 1 - p : p; }
function perspLabel() {
  if (!S.side) return `el equipo ${S.team}`;
  return S.team === S.side ? `tu equipo (${S.side})` : `el equipo contrario (${S.team})`;
}
function relation(team) { return S.side ? (team === S.side ? "a tu favor" : "en tu contra") : ""; }
function nMembers() { return S.res?.modelo?.semillas?.length || 1; }

// toggle de perspectiva: "Equipo contrario / Mi equipo" si se sabe tu lado; si no, "Equipo azul / Equipo rojo"
function renderToggle() {
  const opts = S.side
    ? [{ t: other(S.side), label: "Equipo contrario" }, { t: S.side, label: "Mi equipo" }]
    : [{ t: "azul", label: "Equipo azul" }, { t: "rojo", label: "Equipo rojo" }];
  $("#persp").innerHTML = opts.map((o) =>
    `<button type="button" data-t="${o.t}" aria-pressed="${o.t === S.team}">${o.label}</button>`).join("");
}

function render() {
  const r = S.res;
  const all = [...r.momentos, ...r.candidatos];
  $("#an-title").innerHTML = `Victoria <span class="${r.gana}">lado ${r.gana}</span>`;
  $("#an-side").innerHTML = S.side ? `<span class="badge neutral">Tu equipo: ${esc(S.side)}</span>` : "";
  const fuente = { riot: "descargada de Riot", raw_cache: "datos guardados", memoria: "caché", disco: "caché", demo: "partida de ejemplo" }[r.fuente] || r.fuente;
  $("#an-meta").textContent = `${r.match_id} · ${r.duracion_min} min · parche ${r.parche} · ${fuente}`;
  if (S.models.length) $("#model").value = r.modelo.nombre;
  const note = modelNote(r);
  $("#model-note").innerHTML = note;
  show($("#model-note"), !!note);
  $("#an-warnings").innerHTML = (r.avisos || []).map((w) => `<div class="msg">${esc(w)}</div>`).join("");

  renderToggle();
  // leyenda: la forma del triángulo depende de hacia dónde mueve la curva en esta perspectiva
  $("#lg-tri-azul").className = `tri ${flipped() ? "down" : "up"} azul`;
  $("#lg-tri-rojo").className = `tri ${flipped() ? "up" : "down"} rojo`;
  show($("#lg-band"), nMembers() > 1);
  $("#lg-band-t").textContent = `Rango entre ${nMembers()} entrenamientos`;

  // fichas
  $("#t-mom").textContent = r.momentos.length;
  $("#t-mom-s").textContent = r.candidatos.length ? `+ ${r.candidatos.length} posibles` : "sin posibles extra";
  const rv = r.revision;
  $("#t-full").textContent = rv ? `${Math.round(rv.completa_min)} min` : "–";
  $("#t-tool").textContent = rv ? `${Math.round(rv.con_clips_min)} min` : "–";
  $("#t-tool-s").textContent = rv ? `${rv.ahorro_pct} % menos (estimado)` : "";
  const pf = r.curva[r.curva.length - 1].p_azul;
  $("#t-final").textContent = pct(view(pf));
  $("#t-final-s").textContent = `prob. de ${perspLabel()} al final`.replace("de el ", "del ");

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
  if (!b || !S.res || b.dataset.t === S.team) return;
  S.team = b.dataset.t;
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
      <div class="body">
        <div class="when">Min ${m.desde_min}–${m.minuto}</div>
        <div class="txt">${esc(m.texto.replace(/^min \d+–\d+: /, "").replace(/ \([^()]*\)$/, ""))}</div>
        <div class="evs">${eventsLine(m.eventos)}</div>
        <div class="why" aria-label="Qué explica el cambio">${why}</div>
        <div class="replay">En la repetición: de <code>${mmss(m.desde_min)}</code> a <code>${mmss(m.minuto)}</code></div>
      </div>
      <div class="side">
        <div class="delta ${m.favorece}">${shape} ${Math.abs(Math.round(m.delta_pts))} pts</div>
        <div class="badge ${m.favorece}">a favor de ${m.favorece}</div>
        ${rel ? `<div class="rel">${rel}</div>` : ""}
        <div class="conf">confianza ${esc(m.confianza)}${nMembers() > 1 ? `<br>${m.semillas_de_acuerdo}/${nMembers()} entrenamientos` : ""}</div>
      </div>`;
    const act = () => focusMoment(m.minuto, true);
    el.addEventListener("click", act);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } });
    box.appendChild(el);
  });
}

function setActivePoint(idx) {
  const ch = S.chart;
  const pt = ch?.getDatasetMeta(2).data[idx];
  if (!pt) return null;
  ch.setActiveElements([{ datasetIndex: 2, index: idx }]);
  ch.tooltip.setActiveElements([{ datasetIndex: 2, index: idx }], { x: pt.x, y: pt.y });
  ch.update();
  return pt;
}

function focusMoment(minute, fromList) {
  S.active = minute;
  document.querySelectorAll(".moment").forEach((el) => el.classList.toggle("active", +el.dataset.minute === minute));
  if (S.chart) {
    setActivePoint(S.res.curva.findIndex((c) => c.minuto === minute));
    if (fromList) $("#chart-panel").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
  }
  if (!fromList) {
    const card = document.querySelector(`.moment[data-minute="${minute}"]`);
    card?.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
  }
}

// tabla -> gráfica (solo en ese sentido): la fila elegida hace parpadear su punto en la curva
function renderTable(all) {
  const byMin = Object.fromEntries(all.map((m) => [m.minuto, m]));
  const rows = S.res.curva.map((c, i) => {
    const m = byMin[c.minuto];
    const tag = m ? `${m.candidato ? "posible" : "momento"} (${m.favorece})` : "";
    const band = nMembers() > 1 ? `<td>${pct(Math.min(view(c.p_min), view(c.p_max)))}–${pct(Math.max(view(c.p_min), view(c.p_max)))}</td>` : "";
    return `<tr tabindex="0" data-i="${i}"><td>${c.minuto}</td><td>${pct(view(c.p_azul))}</td>${band}<td style="text-align:left">${tag}${c.baja_certeza ? " baja certeza" : ""}</td></tr>`;
  }).join("");
  const bandH = nMembers() > 1 ? `<th>Rango ${nMembers()} entrenamientos</th>` : "";
  $("#curve-table").innerHTML = `<thead><tr><th>Minuto</th><th>P(gana ${esc(perspLabel())})</th>${bandH}<th style="text-align:left">Nota</th></tr></thead><tbody>${rows}</tbody>`;
}
function pickRow(tr) {
  document.querySelectorAll("#curve-table tr.sel").forEach((x) => x.classList.remove("sel"));
  tr.classList.add("sel");
  pulsePoint(+tr.dataset.i);
}
$("#curve-table").addEventListener("click", (e) => { const tr = e.target.closest("tbody tr"); if (tr) pickRow(tr); });
$("#curve-table").addEventListener("keydown", (e) => {
  const tr = e.target.closest("tbody tr");
  if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); pickRow(tr); }
});

function pulsePoint(idx) {
  const ch = S.chart;
  if (!setActivePoint(idx)) return;
  // si la gráfica no se ve completa (encabezado fijo arriba), se lleva a la vista
  const box = $("#chart-box").getBoundingClientRect();
  const top = $("header.top").offsetHeight;
  if (box.top < top || box.bottom > innerHeight) {
    $("#chart-box").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
  }
  S.pulse = { idx, t0: performance.now() };
  cancelAnimationFrame(S.pulseRaf);
  const step = () => {
    if (!S.pulse) return;
    const done = performance.now() - S.pulse.t0 > PULSE_MS;
    if (done) S.pulse = null;
    ch.draw();
    if (!done) S.pulseRaf = requestAnimationFrame(step);
  };
  S.pulseRaf = requestAnimationFrame(step);
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
  g.strokeStyle = `rgba(160,155,140,${ZONE.hatch})`; g.lineWidth = 1.5;
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
    // ventana de cada clip: relleno del color del equipo favorecido + bordes, para que se lea sobre el fondo
    for (const m of [...S.res.momentos, ...S.res.candidatos]) {
      const rgb = m.candidato ? "160,155,140" : (m.favorece === "azul" ? "58,142,230" : "232,80,91");
      const on = S.active === m.minuto;
      const alpha = m.candidato ? (on ? ZONE.candActive : ZONE.cand) : (on ? ZONE.winActive : ZONE.win);
      const x0 = x.getPixelForValue(m.desde_min), x1 = x.getPixelForValue(m.minuto);
      ctx.fillStyle = `rgba(${rgb},${alpha})`;
      ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
      ctx.strokeStyle = `rgba(${rgb},${m.candidato ? 0.5 : ZONE.edge})`; ctx.lineWidth = 1;
      ctx.setLineDash(m.candidato ? [3, 3] : []);
      ctx.beginPath();
      ctx.moveTo(x0 + 0.5, a.top); ctx.lineTo(x0 + 0.5, a.bottom);
      ctx.moveTo(x1 - 0.5, a.top); ctx.lineTo(x1 - 0.5, a.bottom);
      ctx.stroke();
    }
    // 50 %
    ctx.strokeStyle = `rgba(160,155,140,${ZONE.half})`; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(a.left, y.getPixelForValue(0.5)); ctx.lineTo(a.right, y.getPixelForValue(0.5)); ctx.stroke();
    // guía vertical del hover: DEBAJO de la curva y de los marcadores
    const act = chart.tooltip?.getActiveElements?.() || [];
    if (act.length) {
      const px = act[0].element.x;
      ctx.setLineDash([]);
      ctx.strokeStyle = "rgba(200,170,110,0.6)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
    }
    ctx.restore();
  },
  afterDatasetsDraw(chart) {
    const { ctx, chartArea: a, scales: { x, y } } = chart;
    if (!S.res) return;
    ctx.save();
    ctx.font = "700 12px Cinzel, Georgia, serif"; ctx.textAlign = "center"; ctx.fillStyle = C.ink;
    S.res.momentos.forEach((m, i) => {   // número de cada momento junto a su marcador (sin salirse del gráfico)
      const yy = y.getPixelForValue(view(m.p_despues));
      const up = view(m.p_despues) > view(m.p_antes);
      let ty = up ? yy - 16 : yy + 24;
      if (ty > a.bottom - 2) ty = yy - 16;
      if (ty < a.top + 12) ty = yy + 24;
      ctx.fillText(String(i + 1), Math.min(x.getPixelForValue(m.minuto), a.right - 6), ty);
    });
    // parpadeo del punto elegido en la tabla: 3 ondas doradas (estático si se pide reducir movimiento)
    if (S.pulse) {
      const el = chart.getDatasetMeta(2).data[S.pulse.idx];
      if (el) {
        const k = Math.min(1, (performance.now() - S.pulse.t0) / PULSE_MS);
        const phase = (k * 3) % 1;
        const still = reducedMotion();
        ctx.strokeStyle = `rgba(200,170,110,${still ? 0.9 : 1 - phase})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(el.x, el.y, still ? 11 : 6 + 14 * phase, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = C.gold;
        ctx.beginPath(); ctx.arc(el.x, el.y, 5, 0, 2 * Math.PI); ctx.fill();
      }
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
  S.pulse = null;
  // order: los de número MAYOR se dibujan primero (abajo). Banda < curva < marcadores.
  S.chart = new Chart($("#chart"), {
    type: "line",
    data: {
      datasets: [
        { data: lo, order: 3, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false },
        { data: hi, order: 3, borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: "-1", backgroundColor: `rgba(240,230,210,${ZONE.band})` },
        { data: pts("p_azul"), order: 2, borderColor: C.ink, borderWidth: 2, pointRadius: 0, pointHoverRadius: 5,
          pointHoverBackgroundColor: C.ink, pointHoverBorderColor: C.surface, pointHoverBorderWidth: 2, tension: 0.25 },
        { type: "scatter", data: momentPts, order: 0, pointStyle: "triangle", pointRadius: 9, pointHoverRadius: 11,
          pointRotation: (ctx) => (ctx.raw && !upFor(ctx.raw.m) ? 180 : 0),
          backgroundColor: (ctx) => (ctx.raw?.m.favorece === "rojo" ? C.red : C.blue),
          borderColor: C.surface, borderWidth: 2 },
        { type: "scatter", data: candPts, order: 1, pointStyle: "circle", pointRadius: 6, pointHoverRadius: 8,
          backgroundColor: C.surface, borderColor: C.muted, borderWidth: 2 },
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
              return [`P(gana ${perspLabel()}): ${pct(view(c.p_azul))}`].concat(nMembers() > 1 ? [`rango ${nMembers()} entrenamientos: ${pct(a)}–${pct(b)}`] : [])
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
