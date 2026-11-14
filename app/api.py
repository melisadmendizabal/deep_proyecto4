"""
B11 - API de la app (FastAPI).

  GET /health                                  estado: modelo, θ, si hay API key, partidas en caché
  GET /player?riot_id=Nombre%23TAG&platform=na1&count=5
                                               partidas ranked recientes del jugador
  GET /analyze?match_id=NA1_123&model=M6&refresh=false
                                               curva P(gana azul) + banda, momentos y explicaciones
                                               (model: cualquiera de /health -> modelos; M6 por defecto)

Correr (desde la raíz del repo, con el venv activo):
  uvicorn app.api:app --reload --port 8000        ->  http://127.0.0.1:8000/docs

Caché (orden de búsqueda de /analyze):
  1. memoria  2. data/app_cache/analysis/<id>.json  3. app/demo/<id>.json (fallback offline, B13)
  4. si no está: JSON crudo de data/app_cache/raw/ o del dataset (data/raw/<plataforma>/), y si
     tampoco, la API de Riot (el JSON se guarda en data/app_cache/raw/) -> modelo.
Si la API de Riot falla (key expirada, sin red) y la partida está en caché, se devuelve igual
con un aviso. Cada análisis nuevo se registra en data/app_cache/predicciones.jsonl (monitoreo).

Variables de entorno: RIOT_API_KEY (en .env), MODEL_DIR (modelo principal, default models/m4_gru),
MODELS_ROOT (carpeta con los demás modelos exportados, default models/), APP_CACHE_DIR (default data/app_cache).

Modelos: el principal (M6 + campeones) se carga al arrancar; los demás (M0-M5, M4+draft, exportados con
src/export_model.py) se cargan la primera vez que alguien los pide en el selector.
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

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models" / "m6_draft"))   # principal: M6 (results/mejoras.md)
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", ROOT / "models"))
CACHE_DIR = Path(os.environ.get("APP_CACHE_DIR", config.DATA_DIR / "app_cache"))
DEMO_DIR = ROOT / "app" / "demo"
STATIC_DIR = ROOT / "app" / "static"
MATCH_ID_RE = re.compile(r"^[A-Z0-9]{2,5}_\d{6,12}$")

# selector de modelos: etiqueta, descripción y AUC en test (todas las partidas; b5_metricas.csv y
# b6_semillas.csv, semilla 42). Las diferencias de AUC entre M1-M5 NO son significativas (B5/B6).
MODEL_INFO = {
    "M6": ("Transformer + campeones · modelo principal",
           "Red de atención causal que, además de la partida minuto a minuto, conoce los 10 campeones "
           "(draft); curva = promedio de 3 entrenamientos. Es el modelo principal de la app: el de mejor "
           "AUC del proyecto (results/mejoras.md).", 0.821),
    "M4+draft": ("GRU + campeones", "La GRU de M4 más los 10 campeones (draft); curva = promedio de 3 "
                 "entrenamientos. Empata con M6 (results/mejoras.md).", 0.819),
    "M4": ("GRU · modelo original", "Red recurrente: recuerda toda la partida hasta el minuto actual. Fue el "
           "modelo principal del plan (B6-B12); no conoce los campeones.", 0.812),
    "M5": ("TCN", "Red convolucional temporal: mira los últimos ~29 minutos.", 0.813),
    "M3": ("MLP", "Red neuronal sin memoria: solo ve el minuto actual.", 0.812),
    "M2": ("LightGBM · baseline", "Árboles de decisión con las features del minuto + tendencia de oro.", 0.811),
    "M1": ("Logística · baseline", "Regresión logística con todas las features del minuto actual.", 0.810),
    "M0": ("Solo oro · baseline", "Regresión logística con solo la diferencia de oro y el minuto.", 0.794),
}


class State:
    engine: Engine | None = None       # modelo principal (M6 + campeones)
    engines: dict[str, Engine] = {}    # todos los cargados, por nombre (incluye el principal)
    model_dirs: dict[str, Path] = {}   # modelos disponibles para el selector
    models_lock = threading.Lock()
    riot: RiotClient | None = None
    riot_error: str = ""
    lock = threading.Lock()            # el limitador de tasa de RiotClient no es thread-safe
    memory: dict[str, dict] = {}


S = State()


def init(engine: Engine | None = None, riot: RiotClient | None = None) -> None:
    """Carga el modelo y el cliente de Riot (los tests pasan los suyos)."""
    S.engine = engine or Engine(MODEL_DIR)
    S.engines = {S.engine.name: S.engine}
    S.model_dirs = _discover_models()
    if riot is not None:
        S.riot, S.riot_error = riot, ""
    else:
        try:
            S.riot, S.riot_error = RiotClient(), ""
        except RiotAuthError as exc:
            S.riot, S.riot_error = None, str(exc)
    S.memory = {}


def _discover_models() -> dict[str, Path]:
    """Carpetas de MODELS_ROOT con manifest.json (models/m0_oro, m1_logit, ...), por nombre de modelo."""
    out = {}
    for mf in sorted(MODELS_ROOT.glob("*/manifest.json")) if MODELS_ROOT.exists() else []:
        try:
            name = json.loads(mf.read_text(encoding="utf-8"))["name"]
        except (OSError, ValueError, KeyError):
            continue
        out.setdefault(name, mf.parent)
    return out


def _engine(name: str | None) -> Engine:
    """Motor del modelo pedido (el principal si no se pide ninguno). Los demás se cargan al primer uso."""
    name = (name or "").strip()
    if not name or name.upper() == S.engine.name.upper():
        return S.engine
    key = next((k for k in S.model_dirs if k.upper() == name.upper()), None)
    if key is None:
        raise HTTPException(400, f"Modelo '{name}' no disponible. Disponibles: {', '.join(_model_names())}")
    with S.models_lock:
        if key not in S.engines:
            S.engines[key] = Engine(S.model_dirs[key])
    return S.engines[key]


def _model_names() -> list[str]:
    order = list(MODEL_INFO)
    others = [n for n in S.model_dirs if n != S.engine.name]
    return [S.engine.name] + sorted(others, key=lambda n: (order.index(n) if n in order else 99, n))


def _model_list() -> list[dict]:
    out = []
    for n in _model_names():
        label, desc, auc = MODEL_INFO.get(n, (n, "", None))
        out.append({"nombre": n, "etiqueta": label, "descripcion": desc, "auc_test": auc,
                    "principal": n == S.engine.name})
    return out


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
    # también acepta el nombre del archivo de repetición del cliente: "NA1-5676698765.rofl"
    mid = re.sub(r"\.ROFL$", "", match_id.strip().upper()).replace("-", "_")
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


def _cache_key(match_id: str, eng: Engine) -> str:
    return match_id if eng is S.engine else f"{eng.name}/{match_id}"


def _analysis_path(match_id: str, eng: Engine) -> Path:
    """Principal: analysis/<id>.json (como siempre); los demás: analysis/<modelo>/<id>.json."""
    base = CACHE_DIR / "analysis"
    return base / f"{match_id}.json" if eng is S.engine else base / eng.name / f"{match_id}.json"


def _cached(match_id: str, eng: Engine | None = None) -> tuple[dict | None, str]:
    eng = eng or S.engine
    key = _cache_key(match_id, eng)
    if key in S.memory:
        return S.memory[key], "memoria"
    paths = [(_analysis_path(match_id, eng), "disco")]
    if eng is S.engine:                       # las partidas de ejemplo están hechas con el principal
        paths.append((DEMO_DIR / f"{match_id}.json", "demo"))
    for path, src in paths:
        res = _read_json(path)
        # solo sirve si lo calculó ESTE modelo: al cambiar el principal (M4 -> M6), los análisis
        # guardados en analysis/<id>.json y app/demo/ son de otro modelo y se recalculan
        if res is not None and res.get("modelo", {}).get("nombre") == eng.name:
            S.memory[key] = res
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
            "modelos": _model_list() if eng else [],
            "api_riot": S.riot is not None, "api_riot_error": S.riot_error or None,
            "partidas_en_cache": len(list((CACHE_DIR / "analysis").glob("*.json")))
            if (CACHE_DIR / "analysis").exists() else 0,
            "partidas_demo": sorted(p.stem for p in DEMO_DIR.glob("*.json")) if DEMO_DIR.exists() else [],
            "demos": _demo_summaries()}


def _demo_summaries() -> list[dict]:
    """Resumen de cada partida de ejemplo (pestaña "uso sin conexión"): quién ganó, duración..."""
    out = []
    for p in sorted(DEMO_DIR.glob("*.json")) if DEMO_DIR.exists() else []:
        try:
            d = _read_json(p)
            out.append({"match_id": p.stem, "gana": d["gana"], "duracion_min": d["duracion_min"],
                        "parche": d.get("parche", ""), "n_momentos": len(d.get("momentos", []))})
        except (OSError, ValueError, KeyError):
            continue
    return out


@app.get("/analyze")
def analyze(match_id: str = Query(..., description="p. ej. NA1_5676698765"),
            model: str | None = Query(None, description="M0-M6, M4+draft (ver /health); por defecto el principal, M6"),
            refresh: bool = Query(False, description="ignorar el caché y recalcular")):
    t0 = time.perf_counter()
    mid = _norm_match_id(match_id)
    eng = _engine(model)
    if not refresh:
        res, src = _cached(mid, eng)
        if res is not None:
            return {**res, "fuente": src, "revision": review_estimate(res),
                    "tiempos": {**res.get("tiempos", {}), "total_ms": round(1000 * (time.perf_counter() - t0), 1)}}
    try:
        match, timeline, riot_ms = fetch_raw(mid)
    except HTTPException as exc:
        res, src = _cached(mid, eng)
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
        res = eng.analyze(tables, timings={"riot_ms": round(riot_ms, 1)})
    except AnalysisError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    monitor = res.pop("_monitor", None)
    res["tiempos"]["total_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    _write_json(_analysis_path(mid, eng), res)
    S.memory[_cache_key(mid, eng)] = res
    if eng is S.engine:            # el monitoreo vigila el modelo en producción (el principal)
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
        dur_min = info["gameDuration"] / 60
        cs = int(me.get("totalMinionsKilled", 0)) + int(me.get("neutralMinionsKilled", 0))
        vision = me.get("visionScore")
        out.append({"match_id": mid, "jugador_encontrado": found,
                    "campeon": me.get("championName"), "rol": me.get("teamPosition"),
                    "lado": ("azul" if me.get("teamId") == 100 else "rojo") if found else None,
                    "gano": bool(me.get("win")) if found else None,
                    "kda": f"{me.get('kills', 0)}/{me.get('deaths', 0)}/{me.get('assists', 0)}" if found else None,
                    # farm (súbditos + monstruos de jungla) y puntaje de visión, también por minuto
                    "cs": cs if found else None,
                    "cs_min": round(cs / dur_min, 1) if found and dur_min else None,
                    "vision": int(vision) if found and vision is not None else None,
                    "vision_min": round(vision / dur_min, 1) if found and vision is not None and dur_min else None,
                    "duracion_min": round(dur_min, 1),
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
