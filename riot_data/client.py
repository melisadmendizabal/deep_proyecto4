"""
Cliente HTTP para la API de Riot.

Se encarga de tres cosas para que el resto del código no tenga que pensar en ellas:
  1. Autenticación: manda la API key en la cabecera X-Riot-Token.
  2. Límite de tasa: nunca supera N llamadas por ventana de tiempo, por host
     (cada región, p. ej. la1.api.riotgames.com o americas.api.riotgames.com,
     tiene su propio contador).
  3. Errores: reintenta ante 429 (demasiadas llamadas) y 5xx (error del servidor);
     devuelve None ante 404; detiene todo ante 401/403 (key inválida o expirada).
"""
import time
from collections import deque
from urllib.parse import quote

import requests

from . import config


class RiotAuthError(RuntimeError):
    """La API key es inválida o expiró (las development keys duran 24 h)."""


class RateLimiter:
    """Ventanas deslizantes: guarda cuándo se hizo cada llamada y espera si hace falta."""

    def __init__(self, limits):
        self.set_limits(limits)

    def set_limits(self, limits):
        self.limits = list(limits)  # [(max_llamadas, segundos), ...]
        longest = max(seconds for _, seconds in self.limits)
        self.longest = longest
        self.history = getattr(self, "history", deque())

    def acquire(self):
        while True:
            now = time.monotonic()
            while self.history and now - self.history[0] > self.longest:
                self.history.popleft()
            wait = 0.0
            for max_calls, seconds in self.limits:
                in_window = [t for t in self.history if now - t < seconds]
                if len(in_window) >= max_calls:
                    # hay que esperar a que salga la llamada más antigua de esta ventana
                    wait = max(wait, seconds - (now - in_window[0]) + 0.05)
            if wait <= 0:
                self.history.append(now)
                return
            time.sleep(wait)


class RiotClient:
    def __init__(self, api_key: str = config.API_KEY):
        if not api_key:
            raise RiotAuthError(
                "No hay API key. Copien .env.example a .env y peguen su key de "
                "https://developer.riotgames.com"
            )
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = api_key
        self.limiters: dict[str, RateLimiter] = {}
        self.calls = 0

    # ------------------------------------------------------------------ núcleo
    def _limiter(self, host: str) -> RateLimiter:
        if host not in self.limiters:
            self.limiters[host] = RateLimiter(config.RATE_LIMITS)
        return self.limiters[host]

    def _sync_limits(self, host: str, header: str | None):
        """Ajusta los límites a lo que la API dice en X-App-Rate-Limit (ej. '20:1,100:120')."""
        if not header:
            return
        try:
            limits = []
            for part in header.split(","):
                count, seconds = part.split(":")
                limits.append((max(1, int(count) - 1), int(seconds)))  # -1 de margen
            limiter = self._limiter(host)
            if sorted(limits) != sorted(limiter.limits):
                limiter.set_limits(limits)
        except ValueError:
            pass

    def get(self, host_prefix: str, path: str, params: dict | None = None):
        """GET https://{host_prefix}.api.riotgames.com{path}. Devuelve JSON o None si 404."""
        host = f"{host_prefix}.api.riotgames.com"
        url = f"https://{host}{path}"
        for attempt in range(config.MAX_RETRIES):
            self._limiter(host).acquire()
            try:
                resp = self.session.get(url, params=params, timeout=config.REQUEST_TIMEOUT_S)
            except requests.RequestException as exc:
                wait = 2 ** attempt
                print(f"    ! error de red ({exc.__class__.__name__}), reintento en {wait}s")
                time.sleep(wait)
                continue
            self.calls += 1
            self._sync_limits(host, resp.headers.get("X-App-Rate-Limit"))

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None
            if resp.status_code in (401, 403):
                raise RiotAuthError(
                    f"HTTP {resp.status_code}: la API key es inválida o expiró. "
                    "Generen una nueva en https://developer.riotgames.com y actualicen .env"
                )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                kind = resp.headers.get("X-Rate-Limit-Type", "?")
                print(f"    ! 429 límite de tasa ({kind}) en {host}, esperando {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 2 ** attempt
                print(f"    ! HTTP {resp.status_code} en {host}, reintento en {wait}s")
                time.sleep(wait)
                continue
            print(f"    ! HTTP {resp.status_code} en {url}: {resp.text[:200]}")
            return None
        print(f"    ! se agotaron los reintentos para {url}")
        return None

    # --------------------------------------------------------------- endpoints
    # league-v4 (host = plataforma, ej. "la1")
    def apex_league(self, platform: str, tier: str, queue="RANKED_SOLO_5x5"):
        """Lista completa de MASTER / GRANDMASTER / CHALLENGER."""
        name = {"MASTER": "masterleagues", "GRANDMASTER": "grandmasterleagues",
                "CHALLENGER": "challengerleagues"}[tier]
        return self.get(platform, f"/lol/league/v4/{name}/by-queue/{queue}")

    def league_entries(self, platform: str, tier: str, division: str, page: int,
                       queue="RANKED_SOLO_5x5"):
        """Jugadores de IRON..DIAMOND, una página (~205 jugadores) a la vez."""
        return self.get(platform, f"/lol/league/v4/entries/{queue}/{tier}/{division}",
                        params={"page": page})

    # summoner-v4: solo se usa si una entrada de liga no trae puuid (formato antiguo)
    def summoner_by_id(self, platform: str, summoner_id: str):
        return self.get(platform, f"/lol/summoner/v4/summoners/{summoner_id}")

    # account-v1 (host = región macro): Riot ID "Nombre#TAG" -> puuid
    def account_by_riot_id(self, region: str, game_name: str, tag_line: str):
        region = "asia" if region == "sea" else region  # account-v1 no existe en sea
        return self.get(region, "/riot/account/v1/accounts/by-riot-id/"
                                f"{quote(game_name)}/{quote(tag_line)}")

    # match-v5 (host = región macro, ej. "americas")
    def match_ids(self, region: str, puuid: str, count: int, queue: int | None = None,
                  start_time: int | None = None):
        params = {"start": 0, "count": count}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = start_time
        return self.get(region, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params=params)

    def match(self, region: str, match_id: str):
        return self.get(region, f"/lol/match/v5/matches/{match_id}")

    def timeline(self, region: str, match_id: str):
        return self.get(region, f"/lol/match/v5/matches/{match_id}/timeline")
