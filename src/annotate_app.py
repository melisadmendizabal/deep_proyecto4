"""
B8 - Herramienta de anotación de momentos decisivos (Streamlit).

    streamlit run src/annotate_app.py

Muestra SOLO el log de eventos de cada partida (kills, objetivos, torres, marcador), sin la
curva del modelo ni la gráfica de oro, para que la anotación no quede sesgada hacia lo que el
modelo ya ve. Protocolo: docs/protocolo_anotacion.md.

Requiere haber corrido antes:  python src/select_annotation.py
Guarda en annotations/<anotador>.csv (una fila por momento anotado).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402

DATA = config.DATA_DIR / "anotacion"
SEL = ROOT / "annotations" / "seleccion.csv"
TIPOS = ["pelea", "objetivo", "pick", "estructura", "error", "otro"]
COLS = ["anotador", "match_id", "minuto", "tipo", "equipo", "confianza", "nota",
        "sin_momentos", "segundos", "guardado_utc"]
MONSTER = {"DRAGON": "Dragón", "BARON_NASHOR": "Barón", "RIFTHERALD": "Heraldo",
           "HORDE": "Larvas del Vacío", "ATAKHAN": "Atakhan"}
LANE = {"TOP_LANE": "top", "MID_LANE": "mid", "BOT_LANE": "bot"}
TOWER = {"OUTER_TURRET": "exterior", "INNER_TURRET": "interior", "BASE_TURRET": "base",
         "NEXUS_TURRET": "nexo"}
SIDE = {100: "🔵 azul", 200: "🔴 rojo"}


@st.cache_data
def load():
    if not SEL.exists():
        st.error("No existe annotations/seleccion.csv. Corran primero: python src/select_annotation.py")
        st.stop()
    sel = pd.read_csv(SEL)
    m = pd.read_csv(DATA / "matches.csv")
    p = pd.read_csv(DATA / "participants.csv", keep_default_na=False)
    e = pd.read_csv(DATA / "events.csv", low_memory=False)
    return sel, m, p, e


def ann_path(who: str) -> Path:
    return ROOT / "annotations" / f"{who}.csv"


def load_ann(who: str) -> pd.DataFrame:
    path = ann_path(who)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)


def champ_label(pid, players) -> str:
    if pd.isna(pid) or int(pid) == 0:
        return "torre/súbditos"
    r = players.get(int(pid))
    return f"{r['champion_name']} ({r['team_position'].lower() or '?'})" if r is not None else f"#{int(pid)}"


def event_log(match_id: str, e: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    ev = e[e["match_id"] == match_id].sort_values("timestamp_ms")
    pl = p[p["match_id"] == match_id]
    players = {int(r.participant_id): r._asdict() for r in pl.itertuples()}
    team_of = {pid: r["team_id"] for pid, r in players.items()}
    score = {"k": [0, 0], "t": [0, 0], "d": [0, 0]}
    rows, plates = [], {}
    for r in ev.itertuples():
        t = int(r.timestamp_ms) // 1000
        txt, side = None, None
        if r.type == "CHAMPION_KILL":
            victim_team = team_of.get(int(r.victim_id), 0)
            side = 200 if victim_team == 100 else 100
            score["k"][0 if side == 100 else 1] += 1
            asist = str(r.assisting_ids).count(";") + 1 if isinstance(r.assisting_ids, str) and r.assisting_ids else 0
            txt = (f"{champ_label(r.killer_id, players)} mató a {champ_label(r.victim_id, players)}"
                   + (f" (+{asist} asist.)" if asist else ""))
        elif r.type == "CHAMPION_SPECIAL_KILL":
            if r.kill_type == "KILL_ACE":
                side, txt = team_of.get(int(r.killer_id), None), "ACE (equipo rival eliminado)"
            elif r.kill_type == "KILL_MULTI" and r.multi_kill_length >= 3:
                name = {3: "triple", 4: "cuádruple", 5: "PENTAKILL"}.get(int(r.multi_kill_length), "multi")
                side, txt = team_of.get(int(r.killer_id), None), f"{champ_label(r.killer_id, players)}: {name.upper()}"
        elif r.type == "ELITE_MONSTER_KILL":
            side = int(r.killer_team_id) if not pd.isna(r.killer_team_id) else None
            name = MONSTER.get(r.monster_type, str(r.monster_type))
            if r.monster_type == "DRAGON":
                sub = str(r.monster_sub_type).replace("_DRAGON", "").title()
                name = "DRAGÓN ANCESTRAL (elder)" if "ELDER" in str(r.monster_sub_type) else f"Dragón {sub}"
                if side in (100, 200) and "ELDER" not in str(r.monster_sub_type):
                    score["d"][0 if side == 100 else 1] += 1
            elif r.monster_type == "BARON_NASHOR":
                name = "BARÓN"
            txt = name
        elif r.type == "BUILDING_KILL":
            owner = int(r.team_id) if not pd.isna(r.team_id) else 0
            side = 200 if owner == 100 else 100
            if r.building_type == "INHIBITOR_BUILDING":
                txt = f"INHIBIDOR {LANE.get(r.lane_type, '')} destruido"
            else:
                score["t"][0 if side == 100 else 1] += 1
                txt = f"Torre {TOWER.get(r.tower_type, '')} {LANE.get(r.lane_type, '')} destruida"
        elif r.type == "TURRET_PLATE_DESTROYED":
            owner = int(r.team_id) if not pd.isna(r.team_id) else 0
            key = (t // 60, 200 if owner == 100 else 100)
            n, first = plates.get(key, (0, t))
            plates[key] = (n + 1, min(first, t))
            continue
        elif r.type == "GAME_END":
            side, txt = int(r.winning_team), "FIN DE LA PARTIDA"
        if txt:
            rows.append({"min": t // 60, "tiempo": f"{t // 60:02d}:{t % 60:02d}",
                         "equipo": SIDE.get(side, ""), "evento": txt,
                         "kills": f"{score['k'][0]}–{score['k'][1]}",
                         "torres": f"{score['t'][0]}–{score['t'][1]}",
                         "dragones": f"{score['d'][0]}–{score['d'][1]}"})
    for (minute, side), (n, first) in plates.items():
        rows.append({"min": minute, "tiempo": f"{first // 60:02d}:{first % 60:02d}", "equipo": SIDE.get(side, ""),
                     "evento": f"{n} placa(s) de torre", "kills": "", "torres": "", "dragones": ""})
    log = pd.DataFrame(rows)
    if len(log):
        log = log.sort_values(["min", "tiempo"], kind="stable").reset_index(drop=True)
    return log


def main():
    st.set_page_config(page_title="Anotación de momentos", layout="wide")
    sel, m, p, e = load()

    who = st.sidebar.radio("Anotador", ["A", "B"], horizontal=True,
                           help="A = compañero, B = Melisa (como en la guía)")
    mine = sel[sel["anotador"].isin([who, "ambos"])].reset_index(drop=True)
    ann = load_ann(who)
    done = set(ann["match_id"])
    st.sidebar.progress(len(done & set(mine["match_id"])) / max(len(mine), 1),
                        text=f"{len(done & set(mine['match_id']))} de {len(mine)} anotadas")
    labels = [f"{'✓ ' if r.match_id in done else ''}{r.orden}. {r.match_id} · {r.tier.title()} · "
              f"{r.platform.upper()}{' · compartida' if r.anotador == 'ambos' else ''}"
              for r in mine.itertuples()]
    pick = st.sidebar.selectbox("Partida", range(len(mine)), format_func=lambda i: labels[i])
    st.sidebar.markdown("Protocolo: `docs/protocolo_anotacion.md`")
    row = mine.iloc[pick]
    mid = row["match_id"]

    if st.session_state.get("match") != mid:
        st.session_state["match"], st.session_state["t0"] = mid, time.time()

    info = m[m["match_id"] == mid].iloc[0]
    winner = "🔵 azul" if int(info["blue_win"]) == 1 else "🔴 rojo"
    st.title(f"{mid}")
    st.caption(f"{row['tier'].title()} · {row['platform'].upper()} · {int(info['duration_s']) // 60} min · "
               f"ganó **{winner}**{' · terminó por rendición' if int(info['ended_in_surrender']) else ''}")

    pl = p[p["match_id"] == mid]
    comp = {side: " · ".join(f"{r.team_position.lower() or '?'}: {r.champion_name}"
                             for r in pl[pl["team_id"].astype(int) == side].itertuples())
            for side in (100, 200)}
    st.markdown(f"🔵 **Azul:** {comp[100]}  \n🔴 **Rojo:** {comp[200]}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Log de eventos")
        log = event_log(mid, e, p)
        st.dataframe(log.drop(columns="min"), hide_index=True, width="stretch", height=620)

    with right:
        st.subheader("Momentos decisivos (máx. 5)")
        prev = ann[ann["match_id"] == mid]
        base = prev[prev["minuto"].notna()][["minuto", "tipo", "equipo", "confianza", "nota"]]
        if base.empty:
            base = pd.DataFrame({"minuto": [None], "tipo": [None], "equipo": [None],
                                 "confianza": [2], "nota": [""]})
        edited = st.data_editor(
            base.reset_index(drop=True), num_rows="dynamic", width="stretch",
            key=f"ed_{who}_{mid}",
            column_config={
                "minuto": st.column_config.NumberColumn("Minuto", min_value=0, max_value=70, step=1),
                "tipo": st.column_config.SelectboxColumn("Tipo", options=TIPOS),
                "equipo": st.column_config.SelectboxColumn("Beneficia a", options=["azul", "rojo"]),
                "confianza": st.column_config.NumberColumn("Confianza (1–3)", min_value=1, max_value=3, step=1),
                "nota": st.column_config.TextColumn("Nota (qué pasó)"),
            })
        none = st.checkbox("No hay ningún momento claramente decisivo",
                           value=bool(prev["sin_momentos"].any()) if len(prev) else False)
        if st.button("Guardar anotación", type="primary"):
            rows = edited.dropna(subset=["minuto"])
            problems = []
            if len(rows) > 5:
                problems.append("Máximo 5 momentos por partida.")
            if not none and rows.empty:
                problems.append("Anota al menos un momento o marca 'No hay ningún momento claro'.")
            if rows[["tipo", "equipo"]].isna().any().any():
                problems.append("Cada momento necesita tipo y equipo.")
            if problems:
                for msg in problems:
                    st.error(msg)
            else:
                secs = round(time.time() - st.session_state["t0"])
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                new = rows.assign(anotador=who, match_id=mid, sin_momentos=False, segundos=secs,
                                  guardado_utc=now)
                if none:
                    new = pd.DataFrame([{"anotador": who, "match_id": mid, "minuto": None,
                                         "tipo": None, "equipo": None, "confianza": None,
                                         "nota": "", "sin_momentos": True, "segundos": secs,
                                         "guardado_utc": now}])
                out = pd.concat([ann[ann["match_id"] != mid], new[COLS]], ignore_index=True)
                ann_path(who).parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(ann_path(who), index=False)
                st.success(f"Guardado en annotations/{who}.csv ({len(new)} fila(s), {secs} s).")
                st.rerun()


if __name__ == "__main__":
    main()
