"""
B11 - API de la app (FastAPI).

  GET /health                                  estado: modelo, θ, si hay API key, partidas en caché
  GET /player?riot_id=Nombre%23TAG&platform=na1&count=5
                                               partidas ranked recientes del jugador
  GET /analyze?match_id=NA1_123&refresh=false  curva P(gana azul) + banda, momentos y explicaciones

Correr (desde la raíz del repo, con el venv activo):
  uvicorn app.api:app --reload --port 8000        ->  http://127.0.0.1:8000/docs

Caché (orden de búsqueda de /analyze):
  1. memoria  2. data/app_cache/analysis/<id>.json  3. app/demo/<id>.json (fallback offline, B13)
  4. si no está: JSON crudo de data/app_cache/raw/ o del dataset (data/raw/<plataforma>/), y si
     tampoco, la API de Riot (el JSON se guarda en data/app_cache/raw/) -> modelo.
Si la API de Riot falla (key expirada, sin red) y la partida está en caché, se devuelve igual
con un aviso. Cada análisis nuevo se registra en data/app_cache/predicciones.jsonl (monitoreo).

Variables de entorno: RIOT_API_KEY (en .env), MODEL_DIR (default models/m4_gru),
APP_CACHE_DIR (default data/app_cache).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.inference import ROOT, AnalysisError, Engine, platform_of, tables_from_riot
from riot_data import config
from riot_data.client import RiotAuthError, RiotClient

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "m4_gru"))
CACHE_DIR = Path(os.environ.get("APP_CACHE_DIR", config.DATA_DIR / "app_cache"))
DEMO_DIR = ROOT / "app" / "demo"
STATIC_DIR = ROOT / "app" / "static"
MATCH_ID_RE = re.compile(r"^[A-Z0-9]{2,5}_\d{6,12}$")


class State:
    engine: Engine | None = None
    riot: RiotClient | None = None
    riot_error: str = ""
    lock = threading.Lock()            # el limitador de tasa de RiotClient no es thread-safe
    memory: dict[str, dict] = {}


S = State()


def init(engine: Engine | None = None, riot: RiotClient | None = None) -> None:
    """Carga el modelo y el cliente de Riot (los tests pasan los suyos)."""
    S.engine = engine or Engine(MODEL_DIR)
    if riot is not None:
        S.riot, S.riot_error = riot, ""
    else:
        try:
            S.riot, S.riot_error = RiotClient(), ""
        except RiotAuthError as exc:
            S.riot, S.riot_error = None, str(exc)
    S.memory = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if S.engine is None:
        init()
    yield


app = FastAPI(title="Fight Diff · API", version="1.0", lifespan=lifespan,
              description="Probabilidad de victoria minuto a minuto (GRU causal) y momentos "
                          "decisivos de una partida ranked de League of Legends. Solo post-partida.")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _value_params() -> dict:
    """Parámetros del modelo de valor (configs/value.yaml, B10) para estimar el tiempo de revisión."""
    try:
        cfg = yaml.safe_load((ROOT / "configs" / "value.yaml").read_text(encoding="utf-8"))
        return {k: float(cfg[k]["base"]) for k in ("f_rev", "t_setup", "t_clip")}
    except Exception:  # noqa: BLE001 - sin el yaml se usan los valores base documentados
        return {"f_rev": 1.38, "t_setup": 2.0, "t_clip": 2.5}


VALUE = _value_params()


def review_estimate(res: dict) -> dict:
    """Minutos de revisión: partida completa (f_rev × D) vs solo los clips (t_setup + n × t_clip)."""
    n = len(res.get("momentos", []))
    full = VALUE["f_rev"] * res["duracion_min"]
    tool = VALUE["t_setup"] + n * VALUE["t_clip"]
    return {"completa_min": round(full, 1), "con_clips_min": round(tool, 1),
            "ahorro_pct": round(100 * (1 - tool / full), 0) if full else 0, "parametros": VALUE,
            "nota": "Estimación con los parámetros de B10; no incluye el costo de momentos no detectados."}


# ------------------------------------------------------------------ utilidades
def _norm_match_id(match_id: str) -> str:
    mid = match_id.strip().upper()
    if not MATCH_ID_RE.match(mid):
        raise HTTPException(400, f"'{match_id}' no parece un match ID (ejemplo: NA1_5676698765). "
                                 "Si tienes un Riot ID, búscalo en /player.")
    if platform_of(mid) not in config.PLATFORM_TO_REGION:
        raise HTTPException(400, f"Plataforma '{platform_of(mid)}' desconocida. "
                                 f"Válidas: {', '.join(config.PLATFORM_TO_REGION)}")
    return mid


def _region(platform: str) -> str:
    return config.PLATFORM_TO_REGION[platform]


def _read_json(path: Path):
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _riot() -> RiotClient:
    if S.riot is None:
        raise HTTPException(503, f"No hay conexión con la API de Riot: {S.riot_error}")
    return S.riot


def _riot_call(fn, *args, **kwargs):
    with S.lock:
        try:
            return fn(*args, **kwargs)
        except RiotAuthError as exc:
            raise HTTPException(503, str(exc)) from exc


def fetch_raw(match_id: str) -> tuple[dict, dict, float]:
    """(match, timeline, ms gastados en Riot). Usa el caché en disco si existe."""
    raw_dir = CACHE_DIR / "raw"
    pm, pt = raw_dir / f"{match_id}.json.gz", raw_dir / f"{match_id}_timeline.json.gz"
    # caché de la app, y luego los JSON crudos del dataset (data/raw/<plataforma>/, si existen)
    for d in (raw_dir, config.RAW_DIR / platform_of(match_id)):
        match = _read_json(d / f"{match_id}.json.gz")
        timeline = _read_json(d / f"{match_id}_timeline.json.gz")
        if match is not None and timeline is not None:
            return match, timeline, 0.0
    riot = _riot()
    region = _region(platform_of(match_id))
    t0 = time.perf_counter()
    match = _riot_call(riot.match, region, match_id)
    if match is None:
        raise HTTPException(404, f"No se encontró la partida {match_id} en la API de Riot "
                                 "(¿ID correcto? Las partidas muy antiguas ya no están disponibles).")
    timeline = _riot_call(riot.timeline, region, match_id)
    if timeline is None:
        raise HTTPException(404, f"La partida {match_id} no tiene timeline disponible.")
    ms = 1000 * (time.perf_counter() - t0)
    _write_json(pm, match)
    _write_json(pt, timeline)
    return match, timeline, ms


def _cached(match_id: str) -> tuple[dict | None, str]:
    if match_id in S.memory:
        return S.memory[match_id], "memoria"
    for path, src in ((CACHE_DIR / "analysis" / f"{match_id}.json", "disco"),
                      (DEMO_DIR / f"{match_id}.json", "demo")):
        res = _read_json(path)
        if res is not None:
            S.memory[match_id] = res
            return res, src
    return None, ""


def _log_prediction(res: dict, monitor: dict | None) -> None:
    """Una línea por análisis nuevo (docs/monitoreo.md). Sin datos personales: ni nombres ni puuid."""
    rec = {"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "match_id": res["match_id"],
           "plataforma": res["plataforma"], "inicio_utc": res.get("inicio_utc", ""),
           "modelo": res["modelo"]["nombre"], "theta": res["modelo"]["theta"], "parche": res["parche"],
           "duracion_min": res["duracion_min"], "gana": res["gana"],
           "curva_p_azul": [c["p_azul"] for c in res["curva"]],
           "n_momentos": len(res["momentos"]), "n_candidatos": len(res.get("candidatos", [])),
           "features_min15": (monitor or {}).get("features_min15"),
           "tiempos": res["tiempos"], "avisos": res["avisos"]}
    path = CACHE_DIR / "predicciones.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _log_error(match_id: str, msg: str) -> None:
    path = CACHE_DIR / "errores.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                            "match_id": match_id, "error": msg}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    eng = S.engine
    return {"ok": eng is not None, "modelo": eng.name if eng else None,
            "theta": eng.theta if eng else None,
            "semillas": [m.seed for m in eng.members] if eng else [],
            "parches_entrenamiento": eng.manifest.get("train_patches") if eng else [],
            "api_riot": S.riot is not None, "api_riot_error": S.riot_error or None,
            "partidas_en_cache": len(list((CACHE_DIR / "analysis").glob("*.json")))
            if (CACHE_DIR / "analysis").exists() else 0,
            "partidas_demo": sorted(p.stem for p in DEMO_DIR.glob("*.json")) if DEMO_DIR.exists() else []}


@app.get("/analyze")
def analyze(match_id: str = Query(..., description="p. ej. NA1_5676698765"),
            refresh: bool = Query(False, description="ignorar el caché y recalcular")):
    t0 = time.perf_counter()
    mid = _norm_match_id(match_id)
    if not refresh:
        res, src = _cached(mid)
        if res is not None:
            return {**res, "fuente": src, "revision": review_estimate(res),
                    "tiempos": {**res.get("tiempos", {}), "total_ms": round(1000 * (time.perf_counter() - t0), 1)}}
    try:
        match, timeline, riot_ms = fetch_raw(mid)
    except HTTPException as exc:
        res, src = _cached(mid)
        if exc.status_code == 503 and res is not None:  # sin Riot pero con caché: se muestra igual
            return {**res, "fuente": src, "revision": review_estimate(res), "avisos": res.get("avisos", []) + [
                f"Sin conexión con la API de Riot ({exc.detail}); se muestra el análisis guardado."]}
        raise
    try:
        tables = tables_from_riot(match, timeline, platform_of(mid))
    except (KeyError, TypeError, ValueError) as exc:   # Riot cambió el formato de la respuesta
        _log_error(mid, f"formato de Riot inesperado: {exc!r}")
        raise HTTPException(502, "La respuesta de la API de Riot no tiene el formato esperado "
                                 "(¿cambió el esquema de match-v5?). Quedó registrado.") from exc
    try:
        res = S.engine.analyze(tables, timings={"riot_ms": round(riot_ms, 1)})
    except AnalysisError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    monitor = res.pop("_monitor", None)
    res["tiempos"]["total_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    _write_json(CACHE_DIR / "analysis" / f"{mid}.json", res)
    S.memory[mid] = res
    _log_prediction(res, monitor)
    return {**res, "fuente": "riot" if riot_ms else "raw_cache", "revision": review_estimate(res)}


def find_player(match: dict, puuid: str, game_name: str, tag_line: str) -> dict | None:
    """Participante del jugador: por puuid y, si no, por Riot ID (Riot encripta el puuid distinto
    según la API key, así que un JSON descargado con otra key no lo tiene; el Riot ID sí)."""
    parts = match.get("info", {}).get("participants", [])
    hit = next((p for p in parts if p.get("puuid") == puuid), None)
    if hit is None:
        n, t = game_name.casefold(), tag_line.casefold()
        hit = next((p for p in parts if str(p.get("riotIdGameName", "")).casefold() == n
                    and str(p.get("riotIdTagline", "")).casefold() == t), None)
    return hit


@app.get("/player")
def player(riot_id: str = Query(..., description="Nombre#TAG"),
           platform: str = Query("na1", description="na1, la1, euw1, kr..."),
           count: int = Query(5, ge=1, le=10)):
    platform = platform.strip().lower()
    if platform not in config.PLATFORM_TO_REGION:
        raise HTTPException(400, f"Plataforma '{platform}' desconocida. Válidas: {', '.join(config.PLATFORM_TO_REGION)}")
    if "#" not in riot_id:
        raise HTTPException(400, "El Riot ID debe tener la forma Nombre#TAG (por ejemplo Faker#KR1).")
    name, tag = riot_id.strip().rsplit("#", 1)
    region = _region(platform)
    riot = _riot()
    account = _riot_call(riot.account_by_riot_id, region, name, tag)
    if account is None:
        raise HTTPException(404, f"No existe el Riot ID '{riot_id}'.")
    puuid = account["puuid"]
    who = (account.get("gameName", name), account.get("tagLine", tag))
    ids = _riot_call(riot.match_ids, region, puuid, count, queue=config.QUEUE_ID) or []
    out = []
    for mid in ids:
        pm = CACHE_DIR / "raw" / f"{mid}.json.gz"
        match = _read_json(pm)
        me = find_player(match, puuid, *who) if match else None
        if me is None:
            # sin caché, o caché bajado con OTRA API key (Riot encripta el puuid distinto por key y
            # el jugador no aparece por puuid ni por nombre): se descarga de nuevo
            fresh = _riot_call(riot.match, region, mid)
            if fresh is not None:
                match = fresh
                _write_json(pm, match)
                me = find_player(match, puuid, *who)
        if match is None:
            continue
        info = match["info"]
        found = me is not None
        me = me or {}
        out.append({"match_id": mid, "jugador_encontrado": found,
                    "campeon": me.get("championName"), "rol": me.get("teamPosition"),
                    "lado": ("azul" if me.get("teamId") == 100 else "rojo") if found else None,
                    "gano": bool(me.get("win")) if found else None,
                    "kda": f"{me.get('kills', 0)}/{me.get('deaths', 0)}/{me.get('assists', 0)}" if found else None,
                    "duracion_min": round(info["gameDuration"] / 60, 1),
                    "inicio_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(
                        info.get("gameStartTimestamp", info["gameCreation"]) / 1000)),
                    "analizable": info["gameDuration"] >= 15 * 60,
                    "analizada": mid in S.memory or (CACHE_DIR / "analysis" / f"{mid}.json").exists()})
    return {"riot_id": f"{account.get('gameName', name)}#{account.get('tagLine', tag)}",
            "plataforma": platform, "partidas": out}


@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"mensaje": "API de momentos decisivos. Documentación en /docs (el frontend llega en B13)."}
