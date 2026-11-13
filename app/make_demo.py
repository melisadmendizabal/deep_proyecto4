"""
B13 - Fallback offline: precalcula partidas y las guarda en app/demo/<match_id>.json (SÍ va a git).

Si durante la demo falla la API de Riot (key expirada, sin red, límite de tasa), /analyze sirve
estas partidas igual, y el frontend las ofrece como "Partidas de ejemplo".

  python app/make_demo.py                         # las 5 partidas propias de noviembre (default)
  python app/make_demo.py NA1_123 NA1_456         # otras

Los JSON no tienen datos personales: solo el match_id, la curva, los momentos y conteos de eventos
por equipo (ni nombres, ni Riot IDs, ni puuid).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402

import app.api as api  # noqa: E402

DEFAULT = ["NA1_5676698765",   # reservada para la demo (B10)
           "NA1_5676554848",   # gana rojo, 5 momentos
           "NA1_5676575937", "NA1_5676507367", "NA1_5675359401"]
FORBIDDEN = re.compile(r"puuid|riot_?id|gameName|tagLine|summoner", re.I)


def main():
    ids = sys.argv[1:] or DEFAULT
    out = api.DEMO_DIR
    out.mkdir(parents=True, exist_ok=True)
    with TestClient(api.app) as tc:
        for mid in ids:
            r = tc.get("/analyze", params={"match_id": mid})
            if r.status_code != 200:
                print(f"  {mid}: ERROR {r.status_code} {r.json().get('detail')}")
                continue
            res = r.json()
            for k in ("fuente", "revision"):          # se recalculan al servirla
                res.pop(k, None)
            text = json.dumps(res, ensure_ascii=False, indent=1)
            if FORBIDDEN.search(text):
                raise SystemExit(f"{mid}: el JSON tiene un campo con datos personales; no se guarda")
            (out / f"{mid}.json").write_text(text, encoding="utf-8")
            print(f"  {mid}: {res['duracion_min']} min, gana {res['gana']}, {len(res['momentos'])} momentos "
                  f"+ {len(res['candidatos'])} posibles -> {(out / f'{mid}.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
