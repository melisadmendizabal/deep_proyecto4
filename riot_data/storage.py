"""
Escritura en disco.

data/csv/
  la1_gold/        <- un subdirectorio por subconjunto (plataforma_tier)
    matches.csv
    participants.csv
    frames.csv
    events.csv
  na1_master/
    ...
  _skipped.csv     <- partidas descargadas pero descartadas (remakes, muy cortas...)
data/raw/la1/LA1_123.json.gz, LA1_123_timeline.json.gz   (si SAVE_RAW = True)

Los CSV se abren en modo "append": si el script se detiene (Ctrl+C, key expirada,
se cae el internet) y lo vuelven a correr, continúa donde quedó sin repetir partidas.
"""
import csv
import gzip
import json
from pathlib import Path

from . import config
from .parse import TABLE_FIELDS

SKIPPED_FIELDS = ["match_id", "platform", "reason"]


def subset_name(platform: str, tier: str) -> str:
    return f"{platform}_{tier}".lower()


def _append_rows(path: Path, fields: list[str], rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _read_column(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row[column] for row in csv.DictReader(f)}


class Storage:
    def __init__(self, csv_dir: Path = config.CSV_DIR, raw_dir: Path = config.RAW_DIR):
        self.csv_dir = csv_dir
        self.raw_dir = raw_dir
        self.skipped_path = csv_dir / "_skipped.csv"
        self.counts: dict[str, int] = {}   # partidas guardadas por subconjunto
        self.seen: set[str] = set()        # todo match_id ya procesado (guardado o descartado)
        self._load_existing()

    def _load_existing(self):
        if self.csv_dir.exists():
            for sub in self.csv_dir.iterdir():
                if sub.is_dir():
                    ids = _read_column(sub / "matches.csv", "match_id")
                    self.counts[sub.name] = len(ids)
                    self.seen |= ids
        self.seen |= _read_column(self.skipped_path, "match_id")

    def count(self, platform: str, tier: str) -> int:
        return self.counts.get(subset_name(platform, tier), 0)

    def save_match(self, tables: dict[str, list[dict]], platform: str, tier: str):
        sub = self.csv_dir / subset_name(platform, tier)
        # matches.csv se escribe al final: si algo falla a la mitad, la partida no cuenta
        # como guardada y se reintenta (merge.py elimina las filas huérfanas/duplicadas).
        for table in ("participants", "frames", "events", "matches"):
            _append_rows(sub / f"{table}.csv", TABLE_FIELDS[table], tables[table])
        match_id = tables["matches"][0]["match_id"]
        self.seen.add(match_id)
        name = subset_name(platform, tier)
        self.counts[name] = self.counts.get(name, 0) + 1

    def save_skipped(self, match_id: str, platform: str, reason: str):
        _append_rows(self.skipped_path, SKIPPED_FIELDS,
                     [{"match_id": match_id, "platform": platform, "reason": reason}])
        self.seen.add(match_id)

    def save_raw(self, platform: str, match_id: str, match: dict, timeline: dict | None):
        if not config.SAVE_RAW:
            return
        folder = self.raw_dir / platform
        folder.mkdir(parents=True, exist_ok=True)
        with gzip.open(folder / f"{match_id}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(match, f)
        if timeline is not None:
            with gzip.open(folder / f"{match_id}_timeline.json.gz", "wt", encoding="utf-8") as f:
                json.dump(timeline, f)
