"""
Convierte el JSON de match-v5 (detalle + timeline) en filas planas para los CSV.

Cada partida produce:
  - 1 fila en matches.csv       (resumen: quién ganó, parche, duración, bans...)
  - 10 filas en participants.csv (un jugador por fila: campeón, rol, KDA final...)
  - ~10 x minutos filas en frames.csv (estado de cada jugador en cada minuto)
  - N filas en events.csv        (kills, objetivos, torres, wards..., con su timestamp)
"""
import hashlib
from datetime import datetime, timezone

from . import config

MATCH_FIELDS = [
    "match_id", "platform", "region", "tier", "seed_division", "seed_player",
    "patch", "game_version", "game_start_utc", "duration_s", "n_frames",
    "blue_win", "ended_in_surrender", "queue_id", "blue_bans", "red_bans",
]

PARTICIPANT_FIELDS = [
    "match_id", "platform", "tier", "participant_id", "player", "team_id", "win",
    "champion_id", "champion_name", "team_position", "champ_level",
    "kills", "deaths", "assists", "gold_earned", "total_minions_killed",
    "neutral_minions_killed", "damage_to_champions", "vision_score",
]

FRAME_FIELDS = [
    "match_id", "platform", "tier", "minute", "timestamp_ms", "participant_id", "team_id",
    "total_gold", "current_gold", "xp", "level", "minions_killed", "jungle_minions_killed",
    "position_x", "position_y", "damage_to_champions", "damage_taken",
    "time_enemy_spent_controlled",
]

# columna CSV -> campo en el JSON del evento
EVENT_COLUMNS = {
    "killer_id": "killerId",
    "victim_id": "victimId",
    "creator_id": "creatorId",
    "participant_id": "participantId",
    "killer_team_id": "killerTeamId",
    "team_id": "teamId",
    "monster_type": "monsterType",
    "monster_sub_type": "monsterSubType",
    "building_type": "buildingType",
    "tower_type": "towerType",
    "lane_type": "laneType",
    "ward_type": "wardType",
    "kill_type": "killType",
    "multi_kill_length": "multiKillLength",
    "bounty": "bounty",
    "shutdown_bounty": "shutdownBounty",
    "feat_type": "featType",
    "feat_value": "featValue",
    "name": "name",
    "winning_team": "winningTeam",
}
EVENT_FIELDS = (["match_id", "platform", "tier", "minute", "timestamp_ms", "type"]
                + list(EVENT_COLUMNS) + ["assisting_ids", "position_x", "position_y"])


def anonymize(puuid: str) -> str:
    """Hash estable: el mismo jugador siempre da el mismo id, pero no se puede revertir."""
    return hashlib.sha256((config.PUUID_SALT + puuid).encode()).hexdigest()[:16]


def patch_of(game_version: str) -> str:
    """'14.21.628.3416' -> '14.21'"""
    return ".".join(game_version.split(".")[:2])


def is_remake(match: dict) -> bool:
    return any(p.get("gameEndedInEarlySurrender") for p in match["info"]["participants"])


def parse_match(match: dict, timeline: dict, platform: str, tier: str,
                seed_division: str = "", seed_puuid: str = ""):
    """Devuelve un dict {tabla: [filas]} listo para escribirse en CSV."""
    info = match["info"]
    match_id = match["metadata"]["matchId"]
    base = {"match_id": match_id, "platform": platform, "tier": tier}

    teams = {t["teamId"]: t for t in info["teams"]}
    blue = teams.get(100, {})
    red = teams.get(200, {})
    bans = lambda team: ";".join(str(b["championId"]) for b in team.get("bans", []))

    frames_json = timeline["info"]["frames"]
    match_row = {
        **base,
        "region": config.PLATFORM_TO_REGION.get(platform, ""),
        "seed_division": seed_division,
        "seed_player": anonymize(seed_puuid) if seed_puuid else "",
        "patch": patch_of(info["gameVersion"]),
        "game_version": info["gameVersion"],
        "game_start_utc": datetime.fromtimestamp(
            info.get("gameStartTimestamp", info["gameCreation"]) / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": info["gameDuration"],
        "n_frames": len(frames_json),
        "blue_win": int(bool(blue.get("win"))),
        "ended_in_surrender": int(any(p.get("gameEndedInSurrender")
                                      for p in info["participants"])),
        "queue_id": info["queueId"],
        "blue_bans": bans(blue),
        "red_bans": bans(red),
    }

    team_of = {}
    participants = []
    for p in info["participants"]:
        team_of[p["participantId"]] = p["teamId"]
        participants.append({
            **base,
            "participant_id": p["participantId"],
            "player": anonymize(p["puuid"]),
            "team_id": p["teamId"],
            "win": int(bool(p["win"])),
            "champion_id": p["championId"],
            "champion_name": p["championName"],
            "team_position": p.get("teamPosition", ""),
            "champ_level": p.get("champLevel"),
            "kills": p["kills"],
            "deaths": p["deaths"],
            "assists": p["assists"],
            "gold_earned": p["goldEarned"],
            "total_minions_killed": p.get("totalMinionsKilled"),
            "neutral_minions_killed": p.get("neutralMinionsKilled"),
            "damage_to_champions": p.get("totalDamageDealtToChampions"),
            "vision_score": p.get("visionScore"),
        })

    frames, events = [], []
    for minute, frame in enumerate(frames_json):
        for pid_str, pf in frame["participantFrames"].items():
            pid = int(pid_str)
            pos = pf.get("position", {})
            dmg = pf.get("damageStats", {})
            frames.append({
                **base,
                "minute": minute,
                "timestamp_ms": frame["timestamp"],
                "participant_id": pid,
                "team_id": team_of.get(pid, 100 if pid <= 5 else 200),
                "total_gold": pf.get("totalGold"),
                "current_gold": pf.get("currentGold"),
                "xp": pf.get("xp"),
                "level": pf.get("level"),
                "minions_killed": pf.get("minionsKilled"),
                "jungle_minions_killed": pf.get("jungleMinionsKilled"),
                "position_x": pos.get("x"),
                "position_y": pos.get("y"),
                "damage_to_champions": dmg.get("totalDamageDoneToChampions"),
                "damage_taken": dmg.get("totalDamageTaken"),
                "time_enemy_spent_controlled": pf.get("timeEnemySpentControlled"),
            })
        for ev in frame.get("events", []):
            if config.EVENT_TYPES is not None and ev.get("type") not in config.EVENT_TYPES:
                continue
            pos = ev.get("position", {})
            row = {
                **base,
                # minuto en que OCURRIÓ el evento (0 = entre 0:00 y 0:59)
                "minute": ev["timestamp"] // 60000,
                "timestamp_ms": ev["timestamp"],
                "type": ev["type"],
                "assisting_ids": ";".join(map(str, ev.get("assistingParticipantIds", []))),
                "position_x": pos.get("x"),
                "position_y": pos.get("y"),
            }
            for col, key in EVENT_COLUMNS.items():
                row[col] = ev.get(key)
            events.append(row)

    return {
        "matches": [match_row],
        "participants": participants,
        "frames": frames,
        "events": events,
    }


TABLE_FIELDS = {
    "matches": MATCH_FIELDS,
    "participants": PARTICIPANT_FIELDS,
    "frames": FRAME_FIELDS,
    "events": EVENT_FIELDS,
}
