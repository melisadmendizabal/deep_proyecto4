"""
Descarga partidas de la API de Riot y las guarda en CSV, separadas por plataforma y tier.

Ejemplos:
  python collect.py                                    # usa lo definido en riot_data/config.py
  python collect.py --platforms la1 na1 --tiers GOLD MASTER --per-bucket 20
  python collect.py --riot-id "MiNombre#LAN" --platform la1 --count 20   # partidas propias

Se puede detener con Ctrl+C y volver a correr: continúa donde quedó.
"""
import argparse

from riot_data import config
from riot_data.client import RiotAuthError, RiotClient
from riot_data.collector import Collector
from riot_data.storage import Storage


def print_summary(storage: Storage):
    print("\nPartidas guardadas por subconjunto:")
    for name in sorted(storage.counts):
        print(f"  {name:<20} {storage.counts[name]:>6}")
    print(f"  {'TOTAL':<20} {sum(storage.counts.values()):>6}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platforms", nargs="+", default=config.PLATFORMS,
                        help=f"plataformas (default: {' '.join(config.PLATFORMS)})")
    parser.add_argument("--tiers", nargs="+", default=config.TIERS,
                        help=f"tiers (default: {' '.join(config.TIERS)})")
    parser.add_argument("--per-bucket", type=int, default=config.TARGET_PER_BUCKET,
                        help="meta de partidas por plataforma x tier")
    parser.add_argument("--riot-id", help="modo partidas propias: 'Nombre#TAG'")
    parser.add_argument("--platform", default="la1", help="plataforma de la cuenta (--riot-id)")
    parser.add_argument("--count", type=int, default=20, help="cuántas partidas propias (máx 100)")
    args = parser.parse_args()

    platforms = [p.lower() for p in args.platforms]
    tiers = [t.upper() for t in args.tiers]
    unknown = [p for p in platforms + [args.platform.lower()]
               if p not in config.PLATFORM_TO_REGION]
    if unknown:
        parser.error(f"plataforma desconocida: {unknown}. "
                     f"Opciones: {', '.join(config.PLATFORM_TO_REGION)}")

    storage = Storage()
    try:
        collector = Collector(RiotClient(), storage)
        if args.riot_id:
            collector.collect_own(args.riot_id, args.platform.lower(), args.count)
        else:
            print(f"Recolectando {args.per_bucket} partidas x {len(platforms)} plataformas "
                  f"x {len(tiers)} tiers. Ya hay {len(storage.seen)} partidas procesadas.\n")
            collector.collect(platforms, tiers, args.per_bucket)
    except RiotAuthError as exc:
        print(f"\nERROR: {exc}")
    except KeyboardInterrupt:
        print("\nDetenido por el usuario. Lo descargado ya está guardado.")
    print_summary(storage)


if __name__ == "__main__":
    main()
