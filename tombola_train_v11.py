"""
tombola_train_v11.py  ·  Versión 1.1
──────────────────────────────────────
Entrena el modelo v1.1 usando los datos históricos de tombolas.csv.
Al igual que tombola_train.py (v1), recorre automáticamente todos los pares
(sorteo[i] → sorteo[i+1]) sin intervención manual.

Mejoras sobre v1.0:
  1. Pasa el día de la semana (DOW) a cada ronda de predict → scoring más
     preciso durante el entrenamiento (usa el mismo ajuste que en producción).
  2. Usa recompute_learning_v11 (decay temporal) para calcular los factores
     al final → datos recientes pesan más que datos de 2007.
  3. Muestra breakdown por ventana temporal en el resumen final.

USO
───
# Entrenar con todo el historial (primera vez)
python tombola_train_v11.py --sorteo N --reset

# Solo últimos 500 sorteos con verbose cada 100 rondas
python tombola_train_v11.py --sorteo N --reset --last 500 --verbose 100

# Rango de fechas específico
python tombola_train_v11.py --sorteo N --from 2024-01-01 --to 2026-05-23

# Entrenar vespertina
python tombola_train_v11.py --sorteo V --reset

REQUIERE
────────
tombola_N_transitions_v11.json (generado por tombola_analysis_v11.py)
"""

import csv
import json
import math
import argparse
import os
import sys
import re
from collections import defaultdict
from datetime import date, datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tombola_predict_v11 import (
    load_transitions,
    load_learning,
    save_learning,
    compute_scores_v11,
    rank_candidates,
    build_groups,
    recompute_learning_v11,
    NUM_GROUPS,
    GROUP_SIZE,
    BASE_RATE,
    DECAY,
)


# ──────────────────────────────────────────────
# Carga CSV
# ──────────────────────────────────────────────

def parse_date(raw: str) -> date | None:
    cleaned = raw.strip().strip('"').strip("'")
    m = re.match(r"(\d{4}-\d{2}-\d{2})", cleaned)
    return datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None


def load_draws(csv_path: str, sorteo: str) -> list[dict]:
    draws = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sorteo"].strip().upper() != sorteo.upper():
                continue
            d = parse_date(row["date"])
            if d is None:
                continue
            nums = []
            for col in [f"t{i}" for i in range(1, 21)]:
                v = row.get(col, "").strip()
                if v:
                    try:
                        nums.append(f"{int(v):02d}")
                    except ValueError:
                        pass
            if len(nums) == 20:
                draws.append({"date": d, "numbers": nums, "dow": d.weekday()})
    draws.sort(key=lambda x: x["date"])
    return draws


# ──────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────

def train(draws: list[dict], transitions: dict, learning: dict,
          verbose_every: int = 0) -> tuple[dict, int, int, list[int]]:
    """
    Recorre pares (draw[i] → draw[i+1]) y acumula historial.
    - Pasa el DOW real de cada sorteo a compute_scores_v11.
    - Recalcula factores con decay temporal UNA VEZ al final.
    """
    history  = learning["history"]
    factors  = learning["factors"]
    biases   = learning["biases"]

    existing = {(h["date"], h.get("sorteo", "?"))
                for h in history if h.get("actual")}

    new_rounds  = 0
    skipped     = 0
    total_hits  = [0] * NUM_GROUPS

    n = len(draws)
    print(f"\n  Procesando {n - 1} pares...")

    for i in range(n - 1):
        cur  = draws[i]
        nxt  = draws[i + 1]
        key  = (cur["date"].isoformat(), transitions["metadata"]["sorteo"])

        if key in existing:
            skipped += 1
            continue

        # ── Predicción (con DOW del sorteo actual) ─────────────────────────
        scores = compute_scores_v11(
            cur["numbers"], transitions, factors, biases,
            dow=cur["dow"]   # ← NUEVO en v1.1: usa el día de la semana real
        )
        ranked = rank_candidates(scores)
        hl_stats = dict(learning.get("stats", {}))
        hl_stats["__factors__"] = learning.get("factors", factors)
        groups = build_groups(ranked, cur["numbers"], hl_stats)

        pred = {
            f"elite_{j+1}": [num for num, _ in groups[f"elite_{j+1}"]]
            for j in range(NUM_GROUPS)
        }

        # ── Evaluación ──────────────────────────────────────────────────────
        actual_set = set(nxt["numbers"])
        hits_pg    = {}
        hits_cm    = {}
        cumset     = set()
        for j in range(NUM_GROUPS):
            gk = f"elite_{j+1}"
            gs = set(pred[gk])
            hits_pg[gk] = len(actual_set & gs)
            cumset |= gs
            hits_cm[gk] = len(actual_set & cumset)
            total_hits[j] += hits_pg[gk]

        history.append({
            "date":             cur["date"].isoformat(),
            "sorteo":           transitions["metadata"]["sorteo"],
            "dow":              cur["dow"],
            "input":            cur["numbers"],
            "predicted_groups": pred,
            "actual":           nxt["numbers"],
            "hits_per_group":   hits_pg,
            "hits_cumulative":  hits_cm,
        })
        existing.add(key)
        new_rounds += 1

        if verbose_every > 0 and new_rounds % verbose_every == 0:
            acum = hits_cm.get(f"elite_{NUM_GROUPS}", 0)
            avg  = sum(total_hits) / new_rounds
            print(f"    [{new_rounds:>5}]  {cur['date']}  "
                  f"DOW={cur['dow']}  hits_acum={acum}/20  "
                  f"avg/ronda={avg:.2f}")

    # ── Recalcular factores con decay temporal ──────────────────────────────
    print(f"\n  Recalculando factores (decay temporal DECAY={DECAY}) "
          f"desde {len(history)} rondas...")
    new_factors, new_biases, new_stats, new_dow_f, new_dow_b = recompute_learning_v11(history)
    learning["factors"]     = new_factors
    learning["biases"]      = new_biases
    learning["stats"]       = new_stats
    learning["dow_factors"] = new_dow_f
    learning["dow_biases"]  = new_dow_b
    learning["history"]     = history

    return learning, new_rounds, skipped, total_hits


# ──────────────────────────────────────────────
# Resumen
# ──────────────────────────────────────────────

def print_summary(sorteo: str, new_rounds: int, skipped: int,
                  total_hits: list[int], learning: dict):
    s_name    = "Nocturna" if sorteo.upper() == "N" else "Vespertina"
    completed = len([h for h in learning["history"] if h.get("actual")])
    azar_unit = GROUP_SIZE / 100 * 20

    print()
    print("=" * 62)
    print(f"  ENTRENAMIENTO v1.1 COMPLETADO — {s_name} ({sorteo.upper()})")
    print("=" * 62)
    print(f"  Rondas nuevas procesadas  : {new_rounds}")
    print(f"  Rondas ya existentes      : {skipped}")
    print(f"  Total en historial        : {completed}")
    print(f"  Decay temporal aplicado   : {DECAY}  "
          f"(dato más antiguo pesa {math.exp(-DECAY):.0%} del más reciente)")
    print()
    print(f"  Aciertos por grupo ({new_rounds} rondas nuevas):")
    print(f"  {'Grupo':<10} {'Hits':>6} {'Azar':>7} {'Mejora/ronda':>14}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*14}")
    for j in range(NUM_GROUPS):
        gk    = f"elite_{j+1}"
        h     = total_hits[j]
        azar  = round(azar_unit * new_rounds, 1)
        delta = h / max(1, new_rounds) - azar_unit
        print(f"  {gk:<10} {h:>6}  {azar:>6.1f}  {delta:>+13.3f}")

    # Top factores
    stats = learning.get("stats", {})
    reliable = [
        (k, learning["factors"][k], learning["biases"].get(k, 0),
         stats.get(k, {}).get("predicted", 0),
         stats.get(k, {}).get("hits", 0))
        for k in learning["factors"]
        if stats.get(k, {}).get("predicted", 0) >= 10
    ]
    if reliable:
        top = sorted(reliable, key=lambda x: x[1], reverse=True)[:10]
        low = sorted(reliable, key=lambda x: x[1])[:5]
        print()
        print("  Top 10 números con mayor lift aprendido:")
        for k, lift, bias, pred, hits in top:
            prec = hits / pred if pred else 0
            print(f"    {k}  lift={lift:.3f}  bias={bias:+.3f}  "
                  f"precision={prec:.1%}  ({hits}/{pred})")
        print()
        print("  Top 5 penalizados:")
        for k, lift, bias, pred, hits in low:
            prec = hits / pred if pred else 0
            print(f"    {k}  lift={lift:.3f}  bias={bias:+.3f}  "
                  f"precision={prec:.1%}  ({hits}/{pred})")

    print()
    print("  Próximo paso:")
    print(f"    python tombola_predict_v11.py predict --sorteo {sorteo.upper()} \\")
    print(f"      --dow <0=Lun … 4=Vie>  --numbers <20 números>")
    print("=" * 62)
    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrenador v1.1 — Tombola Uruguay (DOW + decay temporal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    parser.add_argument("--csv",
                        default=os.path.join(BASE_DIR, "tombolas.csv"))
    parser.add_argument("--from", dest="date_from", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--to",   dest="date_to",   default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--last", type=int, default=None, metavar="N")
    parser.add_argument("--verbose", type=int, default=0, metavar="CADA_N")
    parser.add_argument("--reset", action="store_true",
                        help="Borrar historial antes de entrenar")
    args = parser.parse_args()

    sorteo = args.sorteo.upper()
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"

    print(f"\n  [tombola_train_v11] Entrenando {s_name} ({sorteo})")
    print(f"  CSV: {args.csv}")

    transitions = load_transitions(sorteo)

    if args.reset:
        learning = {
            "factors":     {f"{n:02d}": 1.0 for n in range(100)},
            "biases":      {f"{n:02d}": 0.0 for n in range(100)},
            "stats":       {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0,
                                         "w_predicted": 0.0, "w_hits": 0.0}
                            for n in range(100)},
            "dow_factors": {str(d): {f"{n:02d}": 1.0 for n in range(100)} for d in range(7)},
            "dow_biases":  {str(d): {f"{n:02d}": 0.0 for n in range(100)} for d in range(7)},
            "history":     [],
        }
        print("  ⚠ Historial reseteado.")
    else:
        learning = load_learning(sorteo)
        existing = len([h for h in learning["history"] if h.get("actual")])
        print(f"  Historial existente: {existing} rondas")

    draws = load_draws(args.csv, sorteo)
    print(f"  Sorteos en CSV: {len(draws)}  ({draws[0]['date']} → {draws[-1]['date']})")

    if args.date_from:
        d = datetime.strptime(args.date_from, "%Y-%m-%d").date()
        draws = [x for x in draws if x["date"] >= d]
    if args.date_to:
        d = datetime.strptime(args.date_to, "%Y-%m-%d").date()
        draws = [x for x in draws if x["date"] <= d]
    if args.last:
        draws = draws[-(args.last + 1):]

    if len(draws) < 2:
        print("ERROR: Se necesitan al menos 2 sorteos."); sys.exit(1)

    print(f"  Rango a procesar: {draws[0]['date']} → {draws[-1]['date']}  "
          f"({len(draws) - 1} pares)")

    learning, new_rounds, skipped, total_hits = train(
        draws, transitions, learning, verbose_every=args.verbose
    )

    save_learning(sorteo, learning)
    print(f"  ✓ Modelo guardado en tombola_{sorteo}_learning_v11.json")

    print_summary(sorteo, new_rounds, skipped, total_hits, learning)


if __name__ == "__main__":
    main()
