"""
tombola_analysis.py
-------------------
Analiza tombolas.csv y genera un JSON de transiciones de frecuencia
entre días consecutivos, filtrable por sorteo (N=Nocturna / V=Vespertina).

Uso:
    python tombola_analysis.py --sorteo N            # Nocturna (por defecto)
    python tombola_analysis.py --sorteo V            # Vespertina
    python tombola_analysis.py --sorteo N --window 365  # Solo últimos 365 días
    python tombola_analysis.py --sorteo N --output mi_archivo.json

Salida JSON:
    {
      "metadata": { ... },
      "transitions": {
        "00": {
          "appearances":    245,          # cuántas veces salió este número
          "days_since_last": 3,           # días desde la última aparición
          "next_day": [                   # qué salió AL DÍA SIGUIENTE
            ["01", 0.3265, 80],          # [numero, porcentaje, conteo_absoluto]
            ...                           # ordenado de mayor a menor probabilidad
          ],
          "recent_next_day": [            # igual pero solo últimos `window` días
            ["01", 0.35, 12],
            ...
          ]
        },
        ...
      }
    }

Mejoras incluidas vs. el formato base:
  1. Conteo absoluto junto al porcentaje (útil para medir confianza).
  2. Análisis reciente (`recent_next_day`) para detectar tendencias actuales.
  3. `days_since_last`: cuántos sorteos (del tipo elegido) han pasado sin que saliera.
  4. Metadata completa: rango de fechas, total de sorteos analizados, timestamp.
  5. Transiciones ordenadas de mayor a menor probabilidad.
  6. Manejo de gaps en el calendario (fines de semana, feriados).
"""

import csv
import json
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, date
import re
import os
import sys


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def parse_date(raw: str) -> date | None:
    """Limpia el campo date del CSV (que viene con comillas extra) y retorna un date."""
    cleaned = raw.strip().strip('"').strip("'")
    # Formato ISO: 2007-01-02T02:00:00.000Z
    match = re.match(r"(\d{4}-\d{2}-\d{2})", cleaned)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    return None


def load_csv(filepath: str, sorteo: str) -> list[dict]:
    """
    Lee el CSV y retorna una lista de sorteos del tipo indicado,
    ordenados por fecha ascendente.
    Cada elemento: { "date": date, "numbers": set[int] }
    """
    draws = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sorteo"].strip().upper() != sorteo.upper():
                continue
            draw_date = parse_date(row["date"])
            if draw_date is None:
                continue
            numbers = set()
            for col in [f"t{i}" for i in range(1, 21)]:
                val = row.get(col, "").strip()
                if val:
                    try:
                        numbers.add(int(val))
                    except ValueError:
                        pass
            if numbers:
                draws.append({"date": draw_date, "numbers": numbers})

    draws.sort(key=lambda x: x["date"])
    return draws


# ─────────────────────────────────────────────
# Núcleo del análisis
# ─────────────────────────────────────────────

def analyze(draws: list[dict], window_days: int | None = None) -> dict:
    """
    Recorre los sorteos y construye:
      - appearances[num]            : cuántas veces salió num
      - transitions[num][next_num]  : cuántas veces next_num salió el día siguiente
                                      (siguiente sorteo del mismo tipo)

    Si `window_days` está definido, también calcula las mismas métricas
    limitadas a los últimos N días del dataset.
    """
    n = len(draws)
    all_nums = range(100)

    appearances = defaultdict(int)          # { num: count }
    transitions = defaultdict(lambda: defaultdict(int))  # { num: { next_num: count } }

    # Para el análisis reciente
    recent_appearances = defaultdict(int)
    recent_transitions = defaultdict(lambda: defaultdict(int))

    # Fecha de corte para análisis reciente
    last_date = draws[-1]["date"] if draws else None
    cutoff = (last_date - timedelta(days=window_days)) if (last_date and window_days) else None

    for i in range(n - 1):
        current = draws[i]
        next_draw = draws[i + 1]

        for num in current["numbers"]:
            appearances[num] += 1
            for next_num in next_draw["numbers"]:
                transitions[num][next_num] += 1

        # Análisis reciente: solo si la fecha ACTUAL está dentro de la ventana
        if cutoff and current["date"] >= cutoff:
            for num in current["numbers"]:
                recent_appearances[num] += 1
                for next_num in next_draw["numbers"]:
                    recent_transitions[num][next_num] += 1

    # El último sorteo suma apariciones pero no tiene "siguiente"
    for num in draws[-1]["numbers"]:
        appearances[num] += 1
        if cutoff and draws[-1]["date"] >= cutoff:
            recent_appearances[num] += 1

    return {
        "appearances": appearances,
        "transitions": transitions,
        "recent_appearances": recent_appearances,
        "recent_transitions": recent_transitions,
    }


def compute_days_since_last(draws: list[dict]) -> dict[int, int | None]:
    """
    Para cada número, calcula cuántos sorteos (no días de calendario)
    han transcurrido desde su última aparición.
    None si nunca apareció.
    """
    last_seen_index: dict[int, int] = {}
    total = len(draws)

    for i, draw in enumerate(draws):
        for num in draw["numbers"]:
            last_seen_index[num] = i

    result = {}
    for num in range(100):
        if num in last_seen_index:
            result[num] = (total - 1) - last_seen_index[num]
        else:
            result[num] = None
    return result


def build_transition_array(num: int, appearances: dict, transitions: dict) -> list:
    """
    Construye el array de transición para `num`:
    [ [next_num_str, pct, count], ... ] ordenado por pct desc.
    pct = count / appearances[num]  (puede ser > 1 si 20 números salen cada día)
    """
    total = appearances.get(num, 0)
    if total == 0:
        return [[f"{n:02d}", 0.0, 0] for n in range(100)]

    result = []
    for next_num in range(100):
        count = transitions[num].get(next_num, 0)
        pct = round(count / total, 4)
        result.append([f"{next_num:02d}", pct, count])

    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ─────────────────────────────────────────────
# Ensamblaje del JSON final
# ─────────────────────────────────────────────

def build_output(draws: list[dict], sorteo: str, window_days: int | None) -> dict:
    stats = analyze(draws, window_days)
    days_since = compute_days_since_last(draws)

    transitions_json = {}
    for num in range(100):
        key = f"{num:02d}"
        next_day = build_transition_array(
            num, stats["appearances"], stats["transitions"]
        )
        recent_next_day = build_transition_array(
            num, stats["recent_appearances"], stats["recent_transitions"]
        ) if window_days else []

        transitions_json[key] = {
            "appearances": stats["appearances"].get(num, 0),
            "days_since_last": days_since.get(num),
            "next_day": next_day,
            **({"recent_next_day": recent_next_day} if window_days else {}),
        }

    first_date = draws[0]["date"].isoformat() if draws else None
    last_date = draws[-1]["date"].isoformat() if draws else None

    output = {
        "metadata": {
            "sorteo": sorteo.upper(),
            "sorteo_name": "Nocturna" if sorteo.upper() == "N" else "Vespertina",
            "total_draws_analyzed": len(draws),
            "date_range": {"from": first_date, "to": last_date},
            "window_days": window_days,
            "numbers_per_draw": 20,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "note": (
                "`next_day[i]` = [numero, pct_historico, conteo]. "
                "pct es conteo / apariciones_del_numero_padre. "
                "Suma de pct ≈ 20 (20 numeros salen cada dia)."
            ),
        },
        "transitions": transitions_json,
    }
    return output


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera JSON de transiciones de frecuencia para tombolas UY."
    )
    parser.add_argument(
        "--sorteo",
        default="N",
        choices=["N", "V", "n", "v"],
        help="Tipo de sorteo: N=Nocturna (default), V=Vespertina",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "tombolas.csv"),
        help="Ruta al archivo CSV (default: tombolas.csv junto al script)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Archivo de salida JSON (default: tombola_<SORTEO>_transitions.json)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Días hacia atrás para el análisis reciente (ej: 365). Opcional.",
    )
    args = parser.parse_args()

    sorteo = args.sorteo.upper()
    output_file = args.output or f"tombola_{sorteo}_transitions.json"

    print(f"[tombola_analysis] Leyendo {args.csv} ...")
    if not os.path.exists(args.csv):
        print(f"ERROR: No se encontró el archivo: {args.csv}", file=sys.stderr)
        sys.exit(1)

    draws = load_csv(args.csv, sorteo)
    print(f"[tombola_analysis] {len(draws)} sorteos {sorteo} encontrados.")

    if not draws:
        print("ERROR: Sin datos para el sorteo indicado.", file=sys.stderr)
        sys.exit(1)

    print(f"[tombola_analysis] Analizando transiciones ...")
    result = build_output(draws, sorteo, args.window)

    # Guardar al mismo directorio que el CSV si no se especificó ruta absoluta
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(args.csv), output_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[tombola_analysis] JSON guardado en: {output_file}")
    print(f"[tombola_analysis] Rango analizado: {result['metadata']['date_range']['from']} → {result['metadata']['date_range']['to']}")


if __name__ == "__main__":
    main()
