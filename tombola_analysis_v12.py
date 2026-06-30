"""
tombola_analysis_v12.py  ·  Versión 1.2
────────────────────────────────────────
Extiende v1.1 con transiciones de multi-salto (gap):
  - next_day2_all: P(num en sorteo[i+2] | num en sorteo[i])  — brecha 2 sorteos
  - next_day3_all: P(num en sorteo[i+3] | num en sorteo[i])  — brecha 3 sorteos

Estas señales capturan patrones de aparición no consecutivos que v1.1 no detecta:
  un número que no salió ayer puede tender a salir pasado mañana.

Mantiene todas las features de v1.1:
  multi-ventana (5s/30d/90d/365d/all), DOW bias, decil bias, co-ocurrencia, days_since_last.

USO
───
python tombola_analysis_v12.py --sorteo N
python tombola_analysis_v12.py --sorteo V
python tombola_analysis_v12.py --sorteo N --output mi_archivo.json

OUTPUT
──────
tombola_N_transitions_v12.json  (o _V_ para vespertina)

Estructura por número (adicional sobre v1.1):
{
  "00": {
    ...todo lo de v1.1 (appearances, days_since_last, dow_bias, decil_bias,
                         next_day_all, next_day_365d, next_day_90d, next_day_30d, next_day_5s)...
    "next_day2_all": [["71", 0.2291, 269], ...],   ← brecha 2 sorteos, historia completa
    "next_day3_all": [["71", 0.2180, 245], ...],   ← brecha 3 sorteos, historia completa
  }
}
"""

import csv
import json
import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ventanas v1.1 (días; None=toda la historia; negativo=últimos N sorteos)
WINDOWS = {"all": None, "365d": 365, "90d": 90, "30d": 30, "5s": -5}

DOW_NAMES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# ── Decil bias ──────────────────────────────────────────────────────────────
DECIL_WINDOW    = 8
DECIL_HOT_THR   = 1.35
DECIL_COLD_THR  = 0.65
DECIL_INFLUENCE = 0.02


# ──────────────────────────────────────────────
# Carga CSV
# ──────────────────────────────────────────────

def parse_date(raw: str) -> date | None:
    cleaned = raw.strip().strip('"').strip("'")
    m = re.match(r"(\d{4}-\d{2}-\d{2})", cleaned)
    return datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None


def load_csv(filepath: str, sorteo: str) -> list[dict]:
    draws = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sorteo"].strip().upper() != sorteo.upper():
                continue
            d = parse_date(row["date"])
            if d is None:
                continue
            nums = set()
            for col in [f"t{i}" for i in range(1, 21)]:
                v = row.get(col, "").strip()
                if v:
                    try:
                        nums.add(int(v))
                    except ValueError:
                        pass
            if nums:
                draws.append({"date": d, "numbers": nums})
    draws.sort(key=lambda x: x["date"])
    return draws


# ──────────────────────────────────────────────
# Transiciones consecutivas (v1.1)
# ──────────────────────────────────────────────

def compute_transitions_for_window(draws: list[dict]) -> tuple[dict, dict]:
    appearances = defaultdict(int)
    transitions = defaultdict(lambda: defaultdict(int))

    for i in range(len(draws) - 1):
        cur = draws[i]["numbers"]
        nxt = draws[i + 1]["numbers"]
        for num in cur:
            appearances[num] += 1
            for nnum in nxt:
                transitions[num][nnum] += 1

    for num in draws[-1]["numbers"]:
        appearances[num] += 1

    return appearances, transitions


def build_next_day_array(num: int, appearances: dict, transitions: dict) -> list:
    total = appearances.get(num, 0)
    if total == 0:
        return [[f"{n:02d}", 0.0, 0] for n in range(100)]
    result = []
    for nxt in range(100):
        count = transitions[num].get(nxt, 0)
        pct   = round(count / total, 4)
        result.append([f"{nxt:02d}", pct, count])
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ──────────────────────────────────────────────
# Transiciones de multi-salto (v1.2 nuevo)
# ──────────────────────────────────────────────

def compute_gap_transitions(draws: list[dict], gap: int) -> tuple[dict, dict]:
    """
    Calcula transiciones de sorteo[i] → sorteo[i+gap].

    Captura patrones de aparición no consecutivos: números que tienden
    a aparecer 2 o 3 sorteos después de un grupo específico.

    Solo cuenta las apariciones de draws[0..n-gap-1] (los que tienen un
    draw futuro en el horizonte `gap`).
    """
    appearances = defaultdict(int)
    transitions = defaultdict(lambda: defaultdict(int))

    n = len(draws)
    for i in range(n - gap):
        cur = draws[i]["numbers"]
        fut = draws[i + gap]["numbers"]
        for num in cur:
            appearances[num] += 1
            for nnum in fut:
                transitions[num][nnum] += 1

    return appearances, transitions


# ──────────────────────────────────────────────
# Sesgo día-de-semana (DOW bias) — igual v1.1
# ──────────────────────────────────────────────

def compute_dow_bias(draws: list[dict]) -> dict[str, list[float]]:
    total_by_dow = [0] * 7
    count_by_dow = defaultdict(lambda: [0] * 7)
    total_all    = 0

    for draw in draws:
        dow = draw["date"].weekday()
        total_by_dow[dow] += 1
        total_all += 1
        for num in draw["numbers"]:
            count_by_dow[num][dow] += 1

    LAPLACE = 1
    dow_bias = {}
    for n in range(100):
        key    = f"{n:02d}"
        counts = count_by_dow.get(n, [0] * 7)
        total_n     = sum(counts)
        global_rate = total_n / max(1, total_all)

        biases = []
        for d in range(7):
            observed = (counts[d] + LAPLACE) / (total_by_dow[d] + LAPLACE * 100)
            expected = global_rate
            bias = round(observed / max(expected, 1e-6), 4) if expected > 0 else 1.0
            biases.append(bias)
        dow_bias[key] = biases

    return dow_bias


# ──────────────────────────────────────────────
# Decil bias — igual v1.1
# ──────────────────────────────────────────────

def compute_decil_bias(draws: list[dict],
                       window: int    = DECIL_WINDOW,
                       hot_thr: float = DECIL_HOT_THR,
                       cold_thr: float= DECIL_COLD_THR,
                       influence: float = DECIL_INFLUENCE) -> dict[str, float]:
    recent   = draws[-window:] if len(draws) >= window else draws
    n_draws  = len(recent)
    expected = n_draws * 20 / 10

    decil_count = [0] * 10
    for draw in recent:
        for num in draw["numbers"]:
            decil_count[num // 10] += 1

    decil_bias = []
    for d in range(10):
        ratio = decil_count[d] / expected if expected > 0 else 1.0
        if ratio < cold_thr:
            coldness = (cold_thr - ratio) / cold_thr
            bias = 1.0 + influence * coldness
        elif ratio > hot_thr:
            hotness = min((ratio - hot_thr) / hot_thr, 1.0)
            bias = 1.0 - influence * hotness * 0.5
        else:
            bias = 1.0
        decil_bias.append(round(bias, 4))

    return {f"{n:02d}": decil_bias[n // 10] for n in range(100)}


# ──────────────────────────────────────────────
# Co-ocurrencia — igual v1.1
# ──────────────────────────────────────────────

def compute_cooccurrence(draws: list[dict], top_n: int = 20) -> dict:
    n_draws     = len(draws)
    appearances = defaultdict(int)
    cocount     = defaultdict(lambda: defaultdict(int))

    for draw in draws:
        nums = list(draw["numbers"])
        for n in nums:
            appearances[n] += 1
        for a in nums:
            for b in nums:
                if a != b:
                    cocount[a][b] += 1

    result = {}
    for i in range(100):
        app_i = appearances.get(i, 0)
        if app_i == 0:
            result[f"{i:02d}"] = []
            continue
        entries = []
        for j in range(100):
            if i == j:
                continue
            app_j = appearances.get(j, 0)
            co    = cocount[i].get(j, 0)
            if app_j == 0:
                continue
            p_j_given_i = co / app_i
            p_j         = app_j / n_draws
            lift = round(p_j_given_i / p_j, 4) if p_j > 0 else 1.0
            entries.append([f"{j:02d}", lift, co])
        entries.sort(key=lambda x: x[1], reverse=True)
        result[f"{i:02d}"] = entries[:top_n]

    return result


# ──────────────────────────────────────────────
# Days since last — igual v1.1
# ──────────────────────────────────────────────

def compute_days_since_last(draws: list[dict]) -> dict[int, int | None]:
    last_seen = {}
    for i, draw in enumerate(draws):
        for num in draw["numbers"]:
            last_seen[num] = i
    total = len(draws)
    return {
        num: (total - 1) - last_seen[num] if num in last_seen else None
        for num in range(100)
    }


# ──────────────────────────────────────────────
# Ensamblaje del JSON final
# ──────────────────────────────────────────────

def build_output(draws: list[dict], sorteo: str) -> dict:
    print(f"  Calculando ventanas de transición (v1.1)...")
    window_data = {}
    last_date = draws[-1]["date"]

    for w_name, w_days in WINDOWS.items():
        if w_days is None:
            window_draws = draws
        elif w_days < 0:
            n = abs(w_days) + 1
            window_draws = draws[-n:] if len(draws) >= n else draws
        else:
            cutoff       = last_date - timedelta(days=w_days)
            window_draws = [d for d in draws if d["date"] >= cutoff]

        if len(window_draws) < 2:
            window_data[w_name] = (defaultdict(int), defaultdict(lambda: defaultdict(int)))
            print(f"    {w_name}: insuficientes datos ({len(window_draws)} sorteos)")
            continue

        app, trans = compute_transitions_for_window(window_draws)
        window_data[w_name] = (app, trans)
        print(f"    {w_name}: {len(window_draws)} sorteos  ({window_draws[0]['date']} → {window_draws[-1]['date']})")

    print(f"  Calculando transiciones multi-salto (v1.2)...")
    gap2_app, gap2_trans = compute_gap_transitions(draws, gap=2)
    gap3_app, gap3_trans = compute_gap_transitions(draws, gap=3)
    print(f"    gap-2: {sum(gap2_app.values()):,} apariciones base  ({len(draws)-2} pares)")
    print(f"    gap-3: {sum(gap3_app.values()):,} apariciones base  ({len(draws)-3} pares)")

    print(f"  Calculando sesgo día-de-semana...")
    dow_bias   = compute_dow_bias(draws)

    print(f"  Calculando días desde última aparición...")
    days_since = compute_days_since_last(draws)

    print(f"  Calculando sesgo de decil (ventana {DECIL_WINDOW} sorteos)...")
    decil_bias = compute_decil_bias(draws)

    print(f"  Calculando co-ocurrencia (pares en mismo sorteo)...")
    cooccur    = compute_cooccurrence(draws)

    # Construir JSON por número
    transitions_json = {}
    for num in range(100):
        key   = f"{num:02d}"
        entry = {
            "appearances":     window_data["all"][0].get(num, 0),
            "days_since_last": days_since.get(num),
            "dow_bias":        dow_bias[key],
            "decil_bias":      decil_bias[key],
        }
        # Ventanas v1.1
        for w_name in WINDOWS:
            app, trans = window_data[w_name]
            entry[f"next_day_{w_name}"] = build_next_day_array(num, app, trans)
        # Multi-salto v1.2
        entry["next_day2_all"] = build_next_day_array(num, gap2_app, gap2_trans)
        entry["next_day3_all"] = build_next_day_array(num, gap3_app, gap3_trans)

        transitions_json[key] = entry

    return {
        "cooccur": cooccur,
        "metadata": {
            "version":       "1.2",
            "sorteo":        sorteo.upper(),
            "sorteo_name":   "Nocturna" if sorteo.upper() == "N" else "Vespertina",
            "total_draws":   len(draws),
            "date_range":    {"from": draws[0]["date"].isoformat(),
                              "to":   draws[-1]["date"].isoformat()},
            "windows":       list(WINDOWS.keys()),
            "gap_windows":   ["next_day2_all", "next_day3_all"],
            "dow_names":     DOW_NAMES,
            "last_updated":  datetime.now().isoformat(timespec="seconds"),
            "note": (
                "v1.2 añade next_day2_all y next_day3_all: transiciones de brecha "
                "2 y 3 sorteos sobre toda la historia."
            ),
        },
        "transitions": transitions_json,
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tombola Analysis v1.2 — transiciones multi-ventana + DOW + gap"
    )
    parser.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    parser.add_argument("--csv",
                        default=os.path.join(BASE_DIR, "tombolas.csv"),
                        help="Ruta al CSV (default: tombolas.csv junto al script)")
    parser.add_argument("--output", default=None,
                        help="Archivo de salida (default: tombola_<SORTEO>_transitions_v12.json)")
    args = parser.parse_args()

    sorteo = args.sorteo.upper()
    output = args.output or os.path.join(
        BASE_DIR, f"tombola_{sorteo}_transitions_v12.json"
    )

    print(f"\n[tombola_analysis_v12] Sorteo: {'Nocturna' if sorteo == 'N' else 'Vespertina'}")
    print(f"  CSV    : {args.csv}")
    print(f"  Output : {output}\n")

    if not os.path.exists(args.csv):
        print(f"ERROR: No se encontró {args.csv}", file=sys.stderr)
        sys.exit(1)

    draws = load_csv(args.csv, sorteo)
    print(f"  Sorteos cargados: {len(draws)}  ({draws[0]['date']} → {draws[-1]['date']})\n")

    result = build_output(draws, sorteo)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ JSON guardado: {output}")
    print(f"  Ventanas v1.1: {', '.join(WINDOWS.keys())}")
    print(f"  Multi-salto v1.2: next_day2_all, next_day3_all")
    print(f"  DOW bias: 7 días de la semana")
    print(f"  Co-ocurrencia: top 20 pares por lift")
    print(f"  Decil bias: ventana {DECIL_WINDOW} sorteos, influence={DECIL_INFLUENCE}\n")


if __name__ == "__main__":
    main()
