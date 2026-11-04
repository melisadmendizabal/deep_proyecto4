"""
Lógica de recolección: de dónde salen las partidas y en qué orden se descargan.

Flujo por cada subconjunto (plataforma x tier), llamado "bucket":
  1. league-v4  -> lista de jugadores de ese tier (los "semillas")
  2. match-v5   -> IDs de las últimas partidas ranked de un jugador semilla
  3. match-v5   -> detalle de la partida  (se descarta si es remake / muy corta)
  4. match-v5   -> timeline de la partida (minuto a minuto)
  5. parse + CSV

Los buckets se recorren en round-robin: una partida de cada uno por turno, alternando
regiones. Así todos crecen parejo y, como cada región tiene su propio límite de
llamadas, mientras una espera las otras siguen avanzando.
"""
import random
import time
from collections import deque
from dataclasses import dataclass, field

from . import config
from .client import RiotClient
from .parse import is_remake, parse_match
from .storage import Storage


class SeedSource:
    """Entrega jugadores (puuid) de un tier/plataforma, sin repetir."""

    def __init__(self, client: RiotClient, platform: str, tier: str, used: set[str]):
        self.client, self.platform, self.tier, self.used = client, platform, tier, used
        self.buffer: list[tuple[dict, str]] = []
        self.exhausted = False
        self.empty_fetches = 0

    def _refill(self):
        if self.tier in config.APEX_TIERS:
            # Una sola llamada trae toda la liga (cientos/miles de jugadores).
            data = self.client.apex_league(self.platform, self.tier)
            entries = (data or {}).get("entries", [])
            self.exhausted = True  # no hay más páginas
            division = "I"
        else:
            division = random.choice(config.DIVISIONS)
            page = random.randint(1, config.MAX_SEED_PAGE)
            entries = self.client.league_entries(self.platform, self.tier, division, page) or []
            if not entries and page != 1:  # la página aleatoria no existía; probar la 1
                entries = self.client.league_entries(self.platform, self.tier, division, 1) or []
            if not entries:
                self.empty_fetches += 1
                self.exhausted = self.empty_fetches >= 8
        random.shuffle(entries)
        self.buffer.extend((e, e.get("rank", division)) for e in entries)

    def next(self) -> tuple[str, str] | None:
        """Devuelve (puuid, división) o None si ya no hay jugadores."""
        while True:
            if not self.buffer:
                if self.exhausted:
                    return None
                self._refill()
                continue
            entry, division = self.buffer.pop()
            puuid = entry.get("puuid")
            if not puuid and entry.get("summonerId"):  # formato antiguo de la API
                summoner = self.client.summoner_by_id(self.platform, entry["summonerId"])
                puuid = (summoner or {}).get("puuid")
            if puuid and puuid not in self.used:
                self.used.add(puuid)
                return puuid, division


@dataclass
class Bucket:
    platform: str
    tier: str
    seeds: SeedSource
    pending: deque = field(default_factory=deque)  # (match_id, division, seed_puuid)
    done: bool = False

    @property
    def region(self) -> str:
        return config.PLATFORM_TO_REGION[self.platform]


class Collector:
    def __init__(self, client: RiotClient, storage: Storage):
        self.client = client
        self.storage = storage
        self.used_seeds: set[str] = set()
        self.queued: set[str] = set()  # IDs en cola en algún bucket (evita duplicar)
        self.start = time.monotonic()

    # ------------------------------------------------------------ una partida
    def process_match(self, platform: str, tier: str, match_id: str,
                      division: str = "", seed_puuid: str = "") -> bool:
        region = config.PLATFORM_TO_REGION[platform]
        match = self.client.match(region, match_id)
        if match is None:
            self.storage.save_skipped(match_id, platform, "not_found")
            return False

        info = match["info"]
        reason = None
        if info.get("queueId") != config.QUEUE_ID:
            reason = f"queue_{info.get('queueId')}"
        elif is_remake(match):
            reason = "remake"
        elif info.get("gameDuration", 0) < config.MIN_DURATION_S:
            reason = "short"
        if reason:
            # se descarta ANTES de pedir el timeline: ahorra una llamada
            self.storage.save_skipped(match_id, platform, reason)
            return False

        timeline = self.client.timeline(region, match_id)
        if timeline is None:
            self.storage.save_skipped(match_id, platform, "no_timeline")
            return False

        tables = parse_match(match, timeline, platform, tier, division, seed_puuid)
        self.storage.save_raw(platform, match_id, match, timeline)
        self.storage.save_match(tables, platform, tier)
        return True

    # ------------------------------------------------------ recolección mixta
    def _fill_pending(self, bucket: Bucket, start_time: int):
        """Pide jugadores semilla hasta conseguir IDs de partidas nuevas."""
        for _ in range(10):
            seed = bucket.seeds.next()
            if seed is None:
                bucket.done = True
                return
            puuid, division = seed
            ids = self.client.match_ids(bucket.region, puuid, config.MATCHES_PER_SEED,
                                        queue=config.QUEUE_ID, start_time=start_time) or []
            for match_id in ids:
                if match_id not in self.storage.seen and match_id not in self.queued:
                    bucket.pending.append((match_id, division, puuid))
                    self.queued.add(match_id)
            if bucket.pending:
                return

    def collect(self, platforms: list[str], tiers: list[str], target: int):
        start_time = int(time.time()) - config.DAYS_BACK * 86400
        # Orden tier -> plataforma para que llamadas consecutivas vayan a regiones distintas.
        buckets = [Bucket(p, t, SeedSource(self.client, p, t, self.used_seeds))
                   for t in tiers for p in platforms]

        def active():
            return [b for b in buckets
                    if not b.done and self.storage.count(b.platform, b.tier) < target]

        while active():
            for bucket in active():
                if not bucket.pending:
                    self._fill_pending(bucket, start_time)
                if not bucket.pending:
                    continue
                match_id, division, seed = bucket.pending.popleft()
                self.queued.discard(match_id)
                if match_id in self.storage.seen:
                    continue
                if self.process_match(bucket.platform, bucket.tier, match_id, division, seed):
                    self._log(bucket.platform, bucket.tier, match_id, target)

        for b in buckets:
            if b.done and self.storage.count(b.platform, b.tier) < target:
                print(f"  aviso: {b.platform}/{b.tier} se quedó sin jugadores semilla")

    # ------------------------------------------------------ partidas propias
    def collect_own(self, riot_id: str, platform: str, count: int):
        """Descarga las últimas partidas ranked de una cuenta (Riot ID 'Nombre#TAG')."""
        if "#" not in riot_id:
            raise ValueError("El Riot ID debe tener la forma Nombre#TAG")
        game_name, tag_line = riot_id.rsplit("#", 1)
        region = config.PLATFORM_TO_REGION[platform]
        account = self.client.account_by_riot_id(region, game_name, tag_line)
        if not account:
            print(f"No se encontró la cuenta {riot_id}")
            return
        ids = self.client.match_ids(region, account["puuid"], count, queue=config.QUEUE_ID) or []
        print(f"{len(ids)} partidas ranked encontradas para {riot_id}")
        for match_id in ids:
            if match_id in self.storage.seen:
                print(f"  {match_id} ya estaba descargada, se omite")
                continue
            if self.process_match(platform, "OWN", match_id, "", account["puuid"]):
                self._log(platform, "OWN", match_id, count)

    # ------------------------------------------------------------------ util
    def _log(self, platform: str, tier: str, match_id: str, target: int):
        elapsed = int(time.monotonic() - self.start)
        total = sum(self.storage.counts.values())
        print(f"[{platform:>4} | {tier:<11}] {self.storage.count(platform, tier):>4}/{target}"
              f"  {match_id:<16}  total={total}  llamadas={self.client.calls}"
              f"  t={elapsed // 60}m{elapsed % 60:02d}s")
