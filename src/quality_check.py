"""
B2 - Detectores de calidad de datos (v0). No borra nada: marca problemas por partida para que
B3 (features) decida qué excluir.

Detectores:
  - Frames incompletos: minutos faltantes o minutos sin exactamente 10 jugadores; n_frames que
    no cuadra con la duración (se esperan ceil(duración/60) + 1 frames, ±1).
  - AFK / leaver: un jugador que, a partir del minuto 4, pasa >= AFK_MIN_RUN minutos seguidos
    SIN ganar XP y en la MISMA posición exacta (quieto en la base = desconectado).
    Validación (muestra de 9.7k partidas): esas partidas terminan en rendición el 41 % de las
    veces vs 24 % en general. Usar solo "sin XP >= 3 min" marcaba 11 % de las partidas con
    rendición normal (24 %): eran jugadores que morían mucho, no AFK. La XP sigue subiendo
    en nivel 18, así que el detector vale para todos los niveles.
  - Ejecuciones: eventos CHAMPION_KILL sin jugador asesino (killer_id = 0: torre, súbditos,
    monstruos). Explican que haya más eventos de kill que kills en participants. Se reporta
    como dato, no como error; sí se marca si las kills de jugadores no cuadran.
  - Sin GAME_END, posición de equipo vacía.

Uso:
  python src/quality_check.py

Salidas:
  results/b2_calidad_partidas.csv   una fila por partida con sus banderas
  results/b2_resumen.md             resumen legible (para la sección 4.2 del informe)

Lee frames.csv y events.csv por trozos para no llenar la RAM (son ~1.7 GB juntos).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from riot_data import config  # noqa: E402

AFK_MIN_RUN = 2          # minutos seguidos sin XP y sin moverse para marcar AFK
AFK_FROM_MINUTE = 4      # antes de esto es normal no ganar XP (salida de base)
CHUNK = 1_000_000
RESULTS = ROOT / "results"
MERGE = config.MERGE_DIR


def iter_complete_matches(path: Path, usecols: list[str]):
    """Lee un CSV ordenado por partida en trozos, entregando solo partidas completas."""
    carry = None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["match_id"].iat[-1]
        carry = chunk[chunk["match_id"] == last]
        done = chunk[chunk["match_id"] != last]
        if len(done):
            yield done
    if carry is not None and len(carry):
        yield carry


def max_run(flag: pd.Series, new_series: pd.Series) -> pd.Series:
    """Largo de la racha de True en curso en cada fila (se reinicia al cambiar de serie)."""
    run_id = ((flag != flag.shift()) | new_series).cumsum()
    return flag.groupby(run_id).cumsum().where(flag, 0)


def frame_checks() -> pd.DataFrame:
    cols = ["match_id", "minute", "participant_id", "xp", "position_x", "position_y"]
    out = []
    for df in iter_complete_matches(MERGE / "frames.csv", cols):
        # integridad de minutos
        per_min = df.groupby(["match_id", "minute"]).size()
        g = per_min.groupby(level="match_id")
        integ = pd.DataFrame({
            "n_minutes": g.size(),
            "max_minute": per_min.reset_index().groupby("match_id")["minute"].max(),
            "bad_minutes": g.apply(lambda s: int((s != 10).sum())),
        })
        integ["missing_minutes"] = integ["max_minute"] + 1 - integ["n_minutes"]

        # AFK
        df = df.sort_values(["match_id", "participant_id", "minute"])
        new_series = ((df["participant_id"] != df["participant_id"].shift())
                      | (df["match_id"] != df["match_id"].shift()))
        late = df["minute"] >= AFK_FROM_MINUTE
        no_xp = (df["xp"].diff() <= 0) & ~new_series & late
        same_pos = ((df["position_x"].diff() == 0) & (df["position_y"].diff() == 0)
                    & ~new_series & late)
        df["xp_run"] = max_run(no_xp, new_series)
        df["afk_run"] = max_run(no_xp & same_pos, new_series)

        per_player = df.groupby(["match_id", "participant_id"]).agg(
            xp_run=("xp_run", "max"), afk_run=("afk_run", "max"))
        afk_rows = df[df["afk_run"] >= AFK_MIN_RUN]
        afk = pd.DataFrame({
            "afk_players": (per_player["afk_run"] >= AFK_MIN_RUN).groupby(level="match_id").sum(),
            "max_minutes_afk": per_player["afk_run"].groupby(level="match_id").max(),
            "max_minutes_without_xp": per_player["xp_run"].groupby(level="match_id").max(),
        })
        # último minuto con XP antes de la primera racha AFK (≈ cuándo se fue)
        afk["afk_first_minute"] = afk_rows.groupby("match_id")["minute"].min() - AFK_MIN_RUN
        out.append(integ.join(afk))
    return pd.concat(out)


def event_checks() -> pd.DataFrame:
    out = []
    for df in iter_complete_matches(MERGE / "events.csv", ["match_id", "type", "killer_id"]):
        kill = df["type"] == "CHAMPION_KILL"
        execution = kill & (df["killer_id"].fillna(0) == 0)
        out.append(pd.DataFrame({
            "kills_events": kill.groupby(df["match_id"]).sum(),
            "executions": execution.groupby(df["match_id"]).sum(),
            "player_kills_events": (kill & ~execution).groupby(df["match_id"]).sum(),
            "has_game_end": (df["type"] == "GAME_END").groupby(df["match_id"]).any(),
            "n_events": df.groupby("match_id").size(),
        }))
    return pd.concat(out)


def proportion_ci(p: float, n: int) -> str:
    half = 1.96 * np.sqrt(p * (1 - p) / n)
    return f"{100 * p:.1f} % (IC 95 %: {100 * (p - half):.1f}–{100 * (p + half):.1f} %)"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    RESULTS.mkdir(exist_ok=True)
    m = pd.read_csv(MERGE / "matches.csv", dtype={"patch": str})
    splits_path = ROOT / "splits" / "splits.csv"
    if splits_path.exists():
        m = m.merge(pd.read_csv(splits_path, usecols=["match_id", "split"]), on="match_id",
                    how="left")
    else:
        print("Aviso: no hay splits/splits.csv todavía (corran primero src/split.py)")
        m["split"] = ""

    parts = pd.read_csv(MERGE / "participants.csv", keep_default_na=False,
                        usecols=["match_id", "kills", "team_position"])
    pq = parts.groupby("match_id").agg(
        kills_participants=("kills", "sum"),
        empty_positions=("team_position", lambda s: int((s == "").sum())))

    print("Revisando frames (integridad y AFK)...")
    fq = frame_checks()
    print("Revisando eventos...")
    eq = event_checks()

    q = (m[["match_id", "split", "platform", "tier", "patch", "duration_s", "n_frames",
            "blue_win", "ended_in_surrender"]]
         .set_index("match_id").join(fq).join(eq).join(pq).reset_index())
    q["duration_min"] = (q["duration_s"] / 60).round(2)
    q["frames_diff"] = q["n_frames"] - (np.ceil(q["duration_s"] / 60) + 1)
    q["afk_players"] = q["afk_players"].fillna(0).astype(int)

    q["flag_frames_incompletos"] = ((q["missing_minutes"] > 0) | (q["bad_minutes"] > 0)
                                    | (q["frames_diff"].abs() > 1))
    q["flag_afk"] = q["afk_players"] > 0
    q["flag_kills_inconsistentes"] = q["player_kills_events"] != q["kills_participants"]
    q["flag_sin_game_end"] = ~q["has_game_end"].fillna(False).astype(bool)
    q["excluir_de_train"] = q["flag_frames_incompletos"] | q["flag_afk"]
    q.to_csv(RESULTS / "b2_calidad_partidas.csv", index=False)

    # ------------------------------------------------------------------ resumen
    n = len(q)
    flags = ["flag_frames_incompletos", "flag_afk", "flag_kills_inconsistentes",
             "flag_sin_game_end", "excluir_de_train"]
    flag_table = pd.DataFrame({
        "bandera": flags,
        "partidas": [int(q[f].sum()) for f in flags],
        "% del total": [f"{100 * q[f].mean():.2f}" for f in flags],
    })
    by_split = (q.groupby("split")[["flag_afk", "excluir_de_train"]].mean().mul(100).round(2)
                .reindex(["train", "val", "test"]).reset_index())
    by_tier = (q.groupby("tier").agg(partidas=("match_id", "size"),
                                     pct_afk=("flag_afk", "mean"),
                                     duracion_media_min=("duration_min", "mean"),
                                     pct_rendicion=("ended_in_surrender", "mean"),
                                     pct_gana_azul=("blue_win", "mean")))
    by_region = (q.groupby("platform").agg(partidas=("match_id", "size"),
                                           pct_afk=("flag_afk", "mean"),
                                           duracion_media_min=("duration_min", "mean"),
                                           pct_rendicion=("ended_in_surrender", "mean"),
                                           pct_gana_azul=("blue_win", "mean")))
    for t in (by_tier, by_region):
        for c in ("pct_afk", "pct_rendicion", "pct_gana_azul"):
            t[c] = (100 * t[c]).round(1)
        t["duracion_media_min"] = t["duracion_media_min"].round(1)
    dur = q["duration_min"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1)
    sizes = pd.read_csv(RESULTS / "b2_split_tamanos.csv") if (
        RESULTS / "b2_split_tamanos.csv").exists() else None
    overlap = pd.read_csv(RESULTS / "b2_fuga_jugadores.csv") if (
        RESULTS / "b2_fuga_jugadores.csv").exists() else None

    md = [
        "# B2 - Split y calidad de datos",
        "",
        "*Generado por `src/split.py` y `src/quality_check.py`. No editar a mano.*",
        "",
        "## Dataset",
        "",
        f"- Partidas: **{n:,}** (24 subconjuntos de plataforma × tier, 1,125 cada uno).",
        f"- Gana el lado azul: {proportion_ci(q['blue_win'].mean(), n)}. "
        "Las clases están balanceadas, así que no hace falta reponderar.",
        f"- Terminan por rendición: {100 * q['ended_in_surrender'].mean():.1f} %.",
        f"- Duración (min): media {dur['mean']}, mediana {dur['50%']}, "
        f"p5 {dur['5%']}, p95 {dur['95%']}, máx {dur['max']}.",
        f"- Parches: {', '.join(f'{k} ({v:,})' for k, v in q['patch'].value_counts().items())}.",
        "",
        "## Split (temporal, estratificado por subconjunto)",
        "",
        "Dentro de cada plataforma × tier: 70 % más antiguo → train, 15 % → val, "
        "15 % más reciente → test. Congelado en `splits/splits.csv`.",
        "",
    ]
    if sizes is not None:
        totals = sizes[["train", "val", "test", "total"]].sum()
        md += [f"Totales: train {totals['train']:,} · val {totals['val']:,} · "
               f"test {totals['test']:,}. Por subconjunto: "
               f"{sizes['train'].iat[0]} / {sizes['val'].iat[0]} / {sizes['test'].iat[0]} "
               "(tabla completa en `results/b2_split_tamanos.csv`).", ""]
    if overlap is not None:
        md += ["### Posible fuga por jugador", "",
               "Partidas de val/test donde aparece al menos un jugador que también está en "
               "train. No se puede evitar del todo, porque los jugadores juegan muchas "
               "partidas; se reporta como limitación.", "", md_table(overlap), ""]
    md += [
        "## Banderas de calidad", "",
        md_table(flag_table), "",
        f"- **AFK:** un jugador ≥ {AFK_MIN_RUN} min seguidos sin ganar XP *y* sin moverse, "
        f"desde el minuto {AFK_FROM_MINUTE}. Rendición en partidas con AFK: "
        f"{100 * q.loc[q['flag_afk'], 'ended_in_surrender'].mean():.1f} % vs "
        f"{100 * q.loc[~q['flag_afk'], 'ended_in_surrender'].mean():.1f} % sin AFK "
        "(validación del detector). Solo \"sin XP ≥ 3 min\" marcaba ~11 % de las partidas sin "
        "más rendiciones de lo normal: eran jugadores que morían mucho, no AFK.",
        f"- **Ejecuciones** (muertes por torre/súbditos/monstruos, sin jugador asesino): "
        f"{int(q['executions'].sum()):,} en total, en "
        f"{100 * (q['executions'] > 0).mean():.1f} % de las partidas. Explican que haya más "
        "eventos CHAMPION_KILL que kills en participants; no es un error.",
        "- **excluir_de_train** = frames incompletos o AFK. Recomendación para B3: excluirlas "
        "de train y reportar val/test con y sin ellas.",
        "- Kills inconsistentes (kills de jugadores en eventos ≠ participants) y la falta de "
        "GAME_END son solo informativas.", "",
        "Por split (% de partidas):", "", md_table(by_split), "",
        "## Por tier", "", md_table(by_tier.reset_index()), "",
        "## Por región", "", md_table(by_region.reset_index()), "",
    ]
    (RESULTS / "b2_resumen.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + flag_table.to_string(index=False))
    print(f"\nGana azul: {proportion_ci(q['blue_win'].mean(), n)}")
    print("\nGuardado: results/b2_calidad_partidas.csv y results/b2_resumen.md")


if __name__ == "__main__":
    main()
