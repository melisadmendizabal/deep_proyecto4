"""
B4 - Análisis exploratorio (sección 4.2 del informe).

Uso:  python src/eda.py        (~2-4 min; lee datos_merge por trozos)

Reglas:
  - Las relaciones con el resultado (qué predice ganar) se calculan SOLO con train, para no
    "mirar" val/test antes de modelar. Los conteos descriptivos usan las 27,000 partidas.
  - Las diferencias de oro/kills/torres por minuto se calculan aquí de forma simple
    (exploratoria). Las features oficiales salen de B3 (src/features.py).

Salidas:
  results/figures/b4_*.png      figuras para el informe
  results/b4_eda_data/*.csv      los datos de cada figura (vista de tabla)
  results/b4_eda.md             reporte con figuras, tablas y hallazgos
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402
from src.quality_check import iter_complete_matches  # noqa: E402

MERGE = config.MERGE_DIR
RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "b4_eda_data"
T = 61  # minutos 0..60

# ---- estilo (paleta de referencia validada: skill dataviz) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categóricos, orden fijo
TIER_ORDER = ["SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"]
TIER_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#1c5cab", "#0d366b"]  # ordinal
REGION_ORDER = ["la1", "na1", "euw1", "kr"]
REGION_NAME = {"la1": "LAN", "na1": "NA", "euw1": "EUW", "kr": "KR"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
})


def save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _wr(rows: list[dict], minute: int, center: float) -> str:
    """% de victoria azul en un bin de la figura oro vs victoria, como texto."""
    for r in rows:
        if r["minuto"] == minute and r["dif_oro_centro"] == center and r["pct_gana_azul"] != "":
            return f"{r['pct_gana_azul']:.0f} %"
    return "s/d"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


# ------------------------------------------------------------------ carga
def load_tables():
    m = pd.read_csv(MERGE / "matches.csv", dtype={"patch": str})
    s = pd.read_csv(ROOT / "splits" / "splits.csv", usecols=["match_id", "split"])
    q = pd.read_csv(RES / "b2_calidad_partidas.csv", usecols=["match_id", "flag_afk"])
    m = m.merge(s, on="match_id").merge(q, on="match_id")
    m = m.reset_index(drop=True)
    m["row"] = np.arange(len(m))
    p = pd.read_csv(MERGE / "participants.csv", keep_default_na=False,
                    usecols=["match_id", "participant_id", "team_id", "champion_name",
                             "team_position", "win"])
    return m, p


def per_minute_arrays(m: pd.DataFrame) -> dict[str, np.ndarray]:
    """Diferencias azul - rojo por partida y minuto [N, 61] (NaN si la partida ya terminó)."""
    row = dict(zip(m["match_id"], m["row"]))
    n = len(m)
    gold = np.full((n, T), np.nan)
    xp = np.full((n, T), np.nan)
    cols = ["match_id", "minute", "team_id", "total_gold", "xp"]
    for df in iter_complete_matches(MERGE / "frames.csv", cols):
        df = df[df["minute"] < T]
        sign = np.where(df["team_id"] == 100, 1, -1)
        g = (df.assign(g=df["total_gold"] * sign, x=df["xp"] * sign)
             .groupby(["match_id", "minute"])[["g", "x"]].sum().reset_index())
        r = g["match_id"].map(row).to_numpy()
        gold[r, g["minute"]] = g["g"]
        xp[r, g["minute"]] = g["x"]

    # eventos: muertes, torres y monstruos, acumulados al minuto (causal: timestamp <= t*60s)
    deaths_blue = np.zeros((n, T)); deaths_red = np.zeros((n, T))      # noqa: E702
    towers_blue = np.zeros((n, T)); towers_red = np.zeros((n, T))      # noqa: E702
    drag_blue = np.zeros((n, T)); drag_red = np.zeros((n, T))          # noqa: E702
    first = {"dragon": {}, "tower": {}, "baron": {}, "kill": {}}
    cols = ["match_id", "timestamp_ms", "type", "victim_id", "team_id", "killer_team_id",
            "monster_type", "building_type"]
    for df in iter_complete_matches(MERGE / "events.csv", cols):
        df = df[df["type"].isin(["CHAMPION_KILL", "BUILDING_KILL", "ELITE_MONSTER_KILL"])]
        r = df["match_id"].map(row).to_numpy()
        tmin = np.ceil(df["timestamp_ms"].to_numpy() / 60000).astype(int)  # cuenta desde ese frame
        ok = tmin < T
        kill = (df["type"] == "CHAMPION_KILL").to_numpy() & ok
        vic = df["victim_id"].to_numpy()
        np.add.at(deaths_blue, (r[kill & (vic <= 5)], tmin[kill & (vic <= 5)]), 1)
        np.add.at(deaths_red, (r[kill & (vic > 5)], tmin[kill & (vic > 5)]), 1)
        tower = ((df["type"] == "BUILDING_KILL") & (df["building_type"] == "TOWER_BUILDING")).to_numpy() & ok
        owner = df["team_id"].to_numpy()  # dueño de la torre destruida
        np.add.at(towers_blue, (r[tower & (owner == 200)], tmin[tower & (owner == 200)]), 1)
        np.add.at(towers_red, (r[tower & (owner == 100)], tmin[tower & (owner == 100)]), 1)
        dragon = ((df["type"] == "ELITE_MONSTER_KILL") & (df["monster_type"] == "DRAGON")).to_numpy() & ok
        kt = df["killer_team_id"].to_numpy()
        np.add.at(drag_blue, (r[dragon & (kt == 100)], tmin[dragon & (kt == 100)]), 1)
        np.add.at(drag_red, (r[dragon & (kt == 200)], tmin[dragon & (kt == 200)]), 1)
        mins = df["timestamp_ms"] / 60000
        for key, sel in [("dragon", dragon),
                         ("tower", tower),
                         ("baron", (df["monster_type"] == "BARON_NASHOR").to_numpy()),
                         ("kill", kill)]:
            firsts = mins[sel].groupby(df["match_id"][sel]).min()
            first[key].update(firsts.to_dict())

    alive = ~np.isnan(gold)
    cum = lambda a: np.where(alive, np.cumsum(a, axis=1), np.nan)  # noqa: E731
    return {
        "gold": gold, "xp": xp,
        "kills": cum(deaths_red) - cum(deaths_blue),
        "towers": cum(towers_blue) - cum(towers_red),
        "dragons": cum(drag_blue) - cum(drag_red),
        "alive": alive, "first": first,
    }


# ------------------------------------------------------------------ análisis
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    print("Cargando tablas...")
    m, p = load_tables()
    print("Calculando diferencias por minuto (frames + eventos)...")
    A = per_minute_arrays(m)
    train = (m["split"] == "train").to_numpy()
    y = m["blue_win"].to_numpy()
    md = ["# B4 - Análisis exploratorio de datos", "",
          "*Generado por `src/eda.py`. Relaciones con el resultado: solo partidas de **train** "
          f"({train.sum():,}). Conteos descriptivos: las {len(m):,} partidas.*", ""]

    # 1. duración y supervivencia ------------------------------------------------
    dur = m["duration_s"] / 60
    alive_n = A["alive"].sum(0)
    surv = pd.DataFrame({"minuto": np.arange(T), "partidas_vivas": alive_n,
                         "pct": (100 * alive_n / len(m)).round(1)})
    surv.to_csv(TAB / "supervivencia.csv", index=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.hist(dur, bins=np.arange(15, 62, 1), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    a1.axvline(dur.median(), color=INK2, linewidth=1, linestyle="--")
    a1.text(dur.median() + 6, a1.get_ylim()[1] * 0.92, f"mediana {dur.median():.1f} min",
            color=INK2, fontsize=9)
    a1.annotate("rendiciones\na los 15 min", (15.5, 1060), xytext=(15.2, 1420), fontsize=9,
                color=INK2, va="bottom",
                arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    a1.set(title="Duración de las partidas", xlabel="minutos", ylabel="partidas")
    a2.plot(surv["minuto"], surv["pct"], color=SERIES[0])
    for mm in (20, 30, 40):
        v = surv.loc[mm, "pct"]
        a2.plot(mm, v, "o", color=SERIES[0], markersize=5)
        a2.annotate(f"{v:.0f} % sigue en el min {mm}", (mm, v), xytext=(6, 4),
                    textcoords="offset points", fontsize=9, color=INK2)
    a2.set(title="Partidas que siguen vivas", xlabel="minuto", ylabel="% de partidas",
           ylim=(0, 105), xlim=(0, 60))
    save(fig, "b4_duracion_supervivencia")
    md += ["## 1. Duración y supervivencia", "",
           "![duración](figures/b4_duracion_supervivencia.png)", "",
           f"- Duración media {dur.mean():.1f} min (p5 {dur.quantile(.05):.1f}, "
           f"p95 {dur.quantile(.95):.1f}). El mínimo es 15 min porque `collect.py` descarta "
           "remakes y partidas más cortas.",
           f"- El pico en 15–16 min ({int(((dur >= 15) & (dur < 16)).sum()):,} partidas, "
           f"{100 * ((dur >= 15) & (dur < 16)).mean():.1f} %) son **rendiciones en cuanto se "
           "habilita la votación**: partidas que se decidieron muy temprano. Ahí el \"momento "
           "decisivo\" está antes del minuto 15.",
           f"- Solo el {surv.loc[30, 'pct']:.0f} % de las partidas llega al minuto 30 y el "
           f"{surv.loc[40, 'pct']:.0f} % al 40. **Consecuencia para la evaluación:** las métricas "
           "de minutos tardíos se calculan con pocas partidas, y además con partidas *parejas* "
           "(las desbalanceadas terminan antes). Hay que reportar métricas por tramo de minuto "
           "con su número de partidas.", ""]

    # 2. diferencia de oro vs probabilidad de ganar ------------------------------
    bins = np.arange(-10000, 10001, 1000)
    centers = (bins[:-1] + bins[1:]) / 2
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, minute in enumerate([10, 15, 20]):
        g = A["gold"][train, minute]
        ok = ~np.isnan(g)
        idx = np.digitize(g[ok], bins) - 1
        yy = y[train][ok]
        wr, cnt = [], []
        for b in range(len(centers)):
            sel = idx == b
            cnt.append(int(sel.sum()))
            wr.append(100 * yy[sel].mean() if sel.sum() >= 30 else np.nan)
            rows.append({"minuto": minute, "dif_oro_centro": centers[b],
                         "partidas": int(sel.sum()), "pct_gana_azul": round(wr[-1], 1)
                         if not np.isnan(wr[-1]) else ""})
        ax.plot(centers / 1000, wr, color=SERIES[i], marker="o", markersize=4,
                label=f"minuto {minute}")
    ax.axhline(50, color=INK2, linewidth=1, linestyle=":")
    ax.set(title="La misma ventaja de oro pesa más temprano que tarde",
           xlabel="diferencia de oro azul − rojo (miles)", ylabel="% de partidas que gana azul",
           ylim=(0, 100))
    ax.legend(loc="upper left")
    save(fig, "b4_oro_vs_victoria")
    pd.DataFrame(rows).to_csv(TAB / "oro_vs_victoria.csv", index=False)

    # 3. poder predictivo por minuto (AUC de una sola variable) -------------------
    auc_rows = []
    for minute in range(1, 41):
        rec = {"minuto": minute}
        for name in ("gold", "xp", "kills", "towers", "dragons"):
            v = A[name][train, minute]
            ok = ~np.isnan(v)
            rec[name] = roc_auc_score(y[train][ok], v[ok]) if ok.sum() > 200 else np.nan
        rec["partidas"] = int((~np.isnan(A["gold"][train, minute])).sum())
        auc_rows.append(rec)
    auc = pd.DataFrame(auc_rows)
    auc.round(4).to_csv(TAB / "auc_por_minuto.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = {"gold": "diferencia de oro", "kills": "diferencia de kills",
              "towers": "diferencia de torres", "dragons": "diferencia de dragones"}
    for i, key in enumerate(["gold", "kills", "towers", "dragons"]):
        ax.plot(auc["minuto"], auc[key], color=SERIES[i], label=labels[key])
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=":")
    ax.text(40, 0.505, "azar", color=INK2, fontsize=9, ha="right")
    ax.set(title=f"El oro solo llega a AUC {auc['gold'].max():.2f} (min "
                 f"{int(auc.loc[auc['gold'].idxmax(), 'minuto'])}) y luego baja",
           xlabel="minuto", ylabel="AUC (una sola variable)", ylim=(0.45, 0.9), xlim=(1, 40))
    ax.legend(loc="lower right")
    save(fig, "b4_auc_por_minuto")
    a = auc.set_index("minuto")
    md += ["## 2. ¿Qué predice la victoria? (solo train)", "",
           "![oro vs victoria](figures/b4_oro_vs_victoria.png)", "",
           "![auc por minuto](figures/b4_auc_por_minuto.png)", "",
           f"- **La diferencia de oro sola** ya da AUC {a.loc[10, 'gold']:.3f} en el minuto 10, "
           f"{a.loc[15, 'gold']:.3f} en el 15 y {a.loc[20, 'gold']:.3f} en el 20. Es el baseline "
           "M0: cualquier modelo complejo tiene que superar esto para justificarse.",
           f"- **La misma ventaja de oro pesa más temprano que tarde:** un equipo que va +5.5k "
           f"gana el {_wr(rows, 10, 5500)} de las veces si es el minuto 10, contra "
           f"{_wr(rows, 20, 5500)} si es el minuto 20 (la curva del minuto 10 es la más "
           "empinada). Una ventaja fija es mayor "
           "en proporción cuando hay menos oro total en juego. Implicaciones: (1) el modelo "
           "necesita el minuto como entrada, porque una regla fija \"+2k de oro = X %\" no "
           "sirve para toda la partida; (2) conviene la feature de **oro relativo** "
           "(diferencia / oro total de ambos equipos), ya prevista en B3.",
           f"- Después del minuto ~{int(a['gold'].idxmax())} el AUC del oro **baja** "
           f"({a['gold'].max():.3f} → {a.loc[40, 'gold']:.3f} en el 40): las partidas que siguen "
           "vivas son las parejas, y en partidas largas el oro importa menos que los objetivos "
           "(barón, elder, inhibidores). Esto justifica usar objetivos y un modelo que combine "
           "información, no solo oro.",
           "- La XP se comporta casi igual que el oro (están muy correlacionadas); las torres "
           "ganan importancia después del minuto 15.", ""]

    # 4. remontadas por tier ------------------------------------------------------
    g15 = A["gold"][:, 15]
    behind = train & ~np.isnan(g15) & (np.abs(g15) >= 3000)
    behind_team_won = np.where(g15 > 0, 1 - y, y)  # 1 = gana el equipo que iba atrás
    cb = (pd.DataFrame({"tier": m["tier"], "remonta": behind_team_won})[behind]
          .groupby("tier")["remonta"].agg(["mean", "size"]).reindex(TIER_ORDER))
    cb["pct"] = 100 * cb["mean"]
    cb["ic95"] = 196 * np.sqrt(cb["mean"] * (1 - cb["mean"]) / cb["size"])
    cb[["size", "pct", "ic95"]].round(1).rename(columns={"size": "partidas"}).to_csv(
        TAB / "remontadas_por_tier.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(TIER_ORDER))
    ax.bar(x, cb["pct"], color=TIER_RAMP, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, cb["pct"], yerr=cb["ic95"], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for xi, v in zip(x, cb["pct"]):
        ax.text(xi, v + cb["ic95"].max() + 1, f"{v:.0f} %", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [t.title() for t in TIER_ORDER])
    ax.set(title="Remontadas: % que gana el equipo que iba ≥ 3k de oro abajo en el min 15",
           ylabel="% de remontadas", ylim=(0, max(cb["pct"] + cb["ic95"]) + 8))
    ax.grid(axis="x", visible=False)
    save(fig, "b4_remontadas_por_tier")
    md += ["## 3. Remontadas por tier (solo train)", "",
           "![remontadas](figures/b4_remontadas_por_tier.png)", "",
           md_table(cb[["size", "pct", "ic95"]].round(1).reset_index()
                    .rename(columns={"size": "partidas", "pct": "% remonta", "ic95": "± IC95"})),
           "",
           f"- Remontar desde −3k en el minuto 15 es raro en todos los tiers "
           f"({cb['pct'].min():.0f}–{cb['pct'].max():.0f} %). Los intervalos de confianza se "
           "solapan, así que **no hay evidencia fuerte de que el elo bajo remonte más** que el "
           "alto. Hipótesis para E3/E5: un modelo entrenado con todos los tiers no debería "
           "descalibrarse mucho por tier; hay que confirmarlo midiendo la calibración por grupo.",
           ""]

    # 5. objetivos: cuándo pasan -------------------------------------------------
    f = A["first"]
    obj = pd.DataFrame({
        "tier": m["tier"],
        "primera_kill": m["match_id"].map(f["kill"]),
        "primer_dragon": m["match_id"].map(f["dragon"]),
        "primera_torre": m["match_id"].map(f["tower"]),
        "primer_baron": m["match_id"].map(f["baron"]),
    })
    timing = obj.drop(columns="tier").describe(percentiles=[.25, .5, .75]).T[
        ["count", "25%", "50%", "75%"]].round(1)
    timing["% de partidas"] = (100 * timing["count"] / len(m)).round(1)
    timing = timing.drop(columns="count").reset_index().rename(columns={"index": "evento"})
    timing.to_csv(TAB / "tiempos_objetivos.csv", index=False)
    by_tier = obj.groupby("tier")[["primer_dragon", "primera_torre", "primer_baron"]].median()
    by_tier = by_tier.reindex(TIER_ORDER).round(1).reset_index()
    md += ["## 4. Cuándo ocurren los objetivos (minuto)", "",
           md_table(timing), "",
           "Mediana por tier:", "", md_table(by_tier), "",
           f"- La primera kill llega muy temprano (mediana {timing.set_index('evento').loc['primera_kill', '50%']} "
           f"min), pero los objetivos grandes tardan: primer dragón ~{timing.set_index('evento').loc['primer_dragon', '50%']}, "
           f"primera torre ~{timing.set_index('evento').loc['primera_torre', '50%']} y barón "
           f"~{timing.set_index('evento').loc['primer_baron', '50%']} (solo en el "
           f"{timing.set_index('evento').loc['primer_baron', '% de partidas']:.0f} % de las partidas). "
           "Antes del minuto ~5 hay poca información: el AUC del oro sigue por debajo de 0.67 "
           "y la app debe marcar esos minutos como \"baja certeza\".",
           f"- El barón se toma antes en elo alto (mediana {by_tier.set_index('tier').loc['SILVER', 'primer_baron']} min en "
           f"Silver vs {by_tier.set_index('tier').loc['MASTER', 'primer_baron']} en Master): los "
           "jugadores de alto elo convierten ventajas en objetivos más rápido.", ""]

    # 6. campeones ----------------------------------------------------------------
    champs = p["champion_name"].value_counts()
    share_top20 = 100 * champs.head(20).sum() / champs.sum()
    rare = int((champs < 100).sum())
    champs.rename("picks").to_csv(TAB / "picks_por_campeon.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(np.arange(1, len(champs) + 1), champs.values, color=SERIES[0])
    ax.fill_between(np.arange(1, len(champs) + 1), champs.values, color=SERIES[0], alpha=0.15,
                    linewidth=0)
    ax.set(title=f"{len(champs)} campeones: pocos muy jugados, cola larga de poco jugados",
           xlabel="campeón (ordenado por popularidad)", ylabel="veces elegido", xlim=(1, len(champs)))
    ax.annotate(f"{champs.index[0]}: {champs.iloc[0]:,}", (1, champs.iloc[0]), xytext=(10, -4),
                textcoords="offset points", fontsize=9, color=INK2)
    save(fig, "b4_campeones")
    md += ["## 5. Campeones y roles", "",
           "![campeones](figures/b4_campeones.png)", "",
           f"- {len(champs)} campeones distintos. Los 20 más jugados suman el {share_top20:.0f} % "
           f"de las elecciones; el menos jugado aparece {champs.min():,} veces "
           f"({rare} campeones con menos de 100).",
           f"- **Implicación para M6 (embeddings de campeones):** todos los campeones tienen al "
           f"menos ~{champs.min():,} apariciones en 27k partidas (unas "
           f"{int(champs.min() * 0.7):,} en train), suficiente para aprender embeddings "
           "pequeños (8–16 dimensiones) con dropout. Aun así, el draft solo suele predecir "
           "poco: M6 es \"Podría\", no prioridad.",
           f"- Roles: {', '.join(f'{k} {v:,}' for k, v in p['team_position'].replace('', 'vacío').value_counts().items())}.",
           ""]

    # 7. lado, región, tier --------------------------------------------------------
    grp = (m.groupby(["platform"]).agg(partidas=("match_id", "size"),
                                       gana_azul=("blue_win", "mean"),
                                       rendicion=("ended_in_surrender", "mean"),
                                       afk=("flag_afk", "mean"),
                                       duracion=("duration_s", "mean"))
           .reindex(REGION_ORDER))
    grp["gana_azul"] = (100 * grp["gana_azul"]).round(1)
    grp["rendicion"] = (100 * grp["rendicion"]).round(1)
    grp["afk"] = (100 * grp["afk"]).round(1)
    grp["duracion"] = (grp["duracion"] / 60).round(1)
    grp = grp.rename(index=REGION_NAME).reset_index().rename(columns={
        "platform": "región", "gana_azul": "% gana azul", "rendicion": "% rendición",
        "afk": "% con AFK", "duracion": "duración media (min)"})
    grp.to_csv(TAB / "por_region.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(grp))
    w = 0.38
    ax.bar(x - w / 2, grp["% rendición"], width=w, color=SERIES[0], label="% rendición",
           edgecolor=SURFACE, linewidth=2)
    ax.bar(x + w / 2, grp["% con AFK"], width=w, color=SERIES[1], label="% con AFK",
           edgecolor=SURFACE, linewidth=2)
    for xi, (r, a_) in enumerate(zip(grp["% rendición"], grp["% con AFK"])):
        ax.text(xi - w / 2, r + 0.6, f"{r:.0f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, a_ + 0.6, f"{a_:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, grp["región"])
    ax.set(title="KR se rinde casi el doble; LAN tiene más AFK", ylabel="% de partidas")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "b4_region")
    md += ["## 6. Diferencias entre regiones", "",
           "![región](figures/b4_region.png)", "", md_table(grp), "",
           "- **No hay sesgo de lado:** azul gana ~50 % en todas las regiones (ver IC en "
           "`results/b2_resumen.md`), así que la etiqueta está balanceada.",
           "- Las regiones se comportan distinto (rendiciones, AFK, duración). Un modelo "
           "entrenado con todo puede calibrar peor en alguna región: es la base de E4 y de la "
           "evaluación por región (E5).", ""]

    # 8. fugas y valores faltantes ------------------------------------------------
    md += ["## 7. Calidad, nulos y fugas de información", "",
           "- Nulos (verificado): `frames.csv` y `matches.csv` no tienen ningún nulo; "
           "`participants.csv` solo tiene 11 `team_position` vacíos (se reemplazan por `UNKNOWN`). "
           "Las columnas vacías de `events.csv` son esperadas (cada tipo de evento usa solo algunas).",
           "- Partidas con AFK (3.3 %) y ejecuciones: ver `results/b2_resumen.md`.",
           f"- **Prueba de cordura de fuga:** en el minuto 1 el AUC del oro es "
           f"{a.loc[1, 'gold']:.3f}: casi azar, como debe ser porque aún no pasa nada. Si un "
           "modelo da AUC alto en los minutos 0–2, hay una fuga.",
           "- Variables que **nunca** deben ser features (contienen el resultado o el futuro): "
           "`duration_s`, `n_frames`, `ended_in_surrender`, `win`, las estadísticas finales de "
           "`participants.csv`, el evento `GAME_END` y cualquier evento posterior al minuto "
           "que se predice. Lista completa en la guía, sección 3.3.", ""]

    (RES / "b4_eda.md").write_text("\n".join(md), encoding="utf-8")
    print("Listo: results/b4_eda.md, results/figures/b4_*.png, results/b4_eda_data/")
    print(auc.set_index("minuto").loc[[1, 5, 10, 15, 20, 25, 30, 35, 40]].round(3).to_string())


if __name__ == "__main__":
    main()
