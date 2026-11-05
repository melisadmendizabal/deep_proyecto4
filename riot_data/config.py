"""
Parámetros de la recolección. Este es el archivo que normalmente van a editar.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_DIR = DATA_DIR / "csv"     # un subdirectorio por subconjunto (plataforma_tier)
RAW_DIR = DATA_DIR / "raw"     # JSON originales comprimidos (respaldo)
MERGE_DIR = DATA_DIR / "datos_merge"  # dataset final mezclado y balanceado (merge.py)


def _load_dotenv(path: Path) -> None:
    """Carga variables de un archivo .env simple (CLAVE=valor) sin dependencias extra."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("RIOT_API_KEY", "")
# Sal para anonimizar los puuid (se guardan como hash, nunca el puuid real).
PUUID_SALT = os.environ.get("PUUID_SALT", "cambia-esta-sal")

# ---------------------------------------------------------------------------
# Qué recolectar
# ---------------------------------------------------------------------------
# Plataformas (servidores). Cada una se intercala con las demás.
PLATFORMS = ["la1", "na1", "euw1", "kr"]

# Tiers. MASTER, GRANDMASTER y CHALLENGER usan un endpoint distinto (lista completa).
TIERS = ["SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"]

# Meta de partidas por subconjunto (plataforma x tier). Total = plataformas x tiers x meta.
# Si la meta es muy alta, el script corre hasta que lo detengan (Ctrl+C) o expire la key.
TARGET_PER_BUCKET = 12000

# merge.py: cuántas partidas al azar tomar de cada subconjunto (clase).
# 1125 x 24 subconjuntos = 27,000 partidas. El resto queda de reserva en data/csv/.
MERGE_PER_SUBSET = 1125

QUEUE_ID = 420                 # 420 = Ranked Solo/Duo
MATCHES_PER_SEED = 5           # partidas recientes que se toman de cada jugador semilla
DAYS_BACK = 30                 # solo partidas de los últimos N días (≈ parches recientes)
MIN_DURATION_S = 15 * 60       # descarta partidas más cortas (remakes / rendiciones muy tempranas)
MAX_SEED_PAGE = 20             # página máxima al pedir jugadores de un tier (IRON..DIAMOND)

# Guardar también el JSON original (.json.gz). Sirve para regenerar los CSV sin volver a
# descargar si después quieren otras columnas. Pónganlo en False para ahorrar disco.
SAVE_RAW = True

# Tipos de evento del timeline que se guardan en events.csv. None = todos.
# Se excluyen por defecto compras de items y subidas de nivel/habilidad (son muchísimos).
EVENT_TYPES = {
    "CHAMPION_KILL",
    "CHAMPION_SPECIAL_KILL",     # first blood, multikills, ace
    "ELITE_MONSTER_KILL",        # dragones, barón, heraldo, larvas, atakhan
    "BUILDING_KILL",             # torres, inhibidores
    "TURRET_PLATE_DESTROYED",
    "DRAGON_SOUL_GIVEN",
    "WARD_PLACED",
    "WARD_KILL",
    "FEAT_UPDATE",
    "OBJECTIVE_BOUNTY_PRESTART",
    "OBJECTIVE_BOUNTY_FINISH",
    "GAME_END",
}

# ---------------------------------------------------------------------------
# Límites de la API
# ---------------------------------------------------------------------------
# Development key: 20 llamadas/1 s y 100 llamadas/120 s POR REGIÓN (host).
# Se dejan un poco por debajo como margen. Si la API responde con otros límites
# (cabecera X-App-Rate-Limit), el cliente se ajusta solo.
RATE_LIMITS = [(19, 1), (95, 120)]
MAX_RETRIES = 5
REQUEST_TIMEOUT_S = 15

# Plataforma -> región "macro" que usa match-v5 y account-v1.
PLATFORM_TO_REGION = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea",
}

APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}
DIVISIONS = ["I", "II", "III", "IV"]
