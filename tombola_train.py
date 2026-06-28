"""
tombola_train.py
────────────────
Entrena el modelo de tombola_predict.py usando los datos históricos del CSV.

En lugar de ejecutar predict + feedback manualmente para cada día, este script
recorre tombolas.csv automáticamente: usa cada sorteo como "entrada" y el
siguiente sorteo del mismo tipo como "resultado real", generando todos los pares
de entrenamiento sin intervención del usuario.

USO
───
# Entrenar con todo el historial (recomendado la primera vez)
python tombola_train.py --sorteo N

# Entrenar con todo el historial vespertino
python tombola_train.py --sorteo V

# Entrenar solo los últimos N sorteos (refuerzo reciente)
python tombola_train.py --sorteo N --last 200

# Entrenar un rango de fechas específico
python tombola_train.py --sorteo N --from 2025-01-01 --to 2026-05-23

# Ver progreso detallado cada 50 rondas
python tombola_train.py --sorteo N --verbose 50

# Resetear el aprendizaje y entrenar desde cero
python tombola_train.py --sorteo N --reset

CÓMO FUNCIONA
─────────────
Para cada par consecutivo de sorteos (día D → día D+1) del tipo elegido:

  1. Se calculan los scores con el modelo actual (transiciones + factores aprendidos)
  2. Se construyen los grupos élite (los 30 mejores números en 5 grupos de 6)
  3. Se comparan con el resultado real del día D+1
  4. Se registra en el historial de aprendizaje
  5. Al final de TODAS las rondas, se recalculan los factores de lift y bias
     una sola vez desde el historial completo (más eficiente que hacerlo ronda a ronda)

RESULTADO
─────────
El archivo tombola_<N|V>_learning.json queda actualizado con los factores
y biases aprendidos. tombola_predict.py los usará automáticamente en el
próximo `predict`.
"""

import csv
import json
import argparse
import os
import sys
import re
from collections import defaultdict
from datetime import date, datetime

# ── Importar funciones de tombola_predict ──────────────────────────────────
# Se importa directamente para evitar overhead de subprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tombola_predict import (
    load_transitions,
    load_learning,
    save_learning,
    compute_scores,
    rank_candidates,
    build_groups,
    recompute_learning_from_history,
    NUM_GROUPS,
    GROUP_SIZE,
)


# ──────────────────────────────────────────────
# Leer CSV
# ──────────────────────────────────────────────

def parse_date(raw: str) -> date | None:
    cleaned = raw.strip().strip('"').strip("'")
    match = re.match(r"(\d{4}-\d{2}-\d{2})", cleaned)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    return None


def load_draws(csv_path: str, sorteo: str) -> list[dict]:
    """
    Lee el CSV y retorna lista de sorteos del tipo indicado, ordenados por fecha.
    Cada elemento: { "date": date, "numbers": list[str] }
    """
    draws = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sorteo"].strip().upper() != sorteo.upper():
                continue
            draw_date = parse_date(row["date"])
            if draw_date is None:
                continue
            numbers = []
            for col in [f"t{i}" for i in range(1, 21)]:
                val = row.get(col, "").strip()
                if val:
                    try:
                        numbers.append(f"{int(val):02d}")
                    except ValueError:
                        pass
            if len(numbers) == 20:
                draws.append({"date": draw_date, "numbers": numbers})

    draws.sort(key=lambda x: x["date"])
    return draws


# ──────────────────────────────────────────────
# Núcleo del entrenamiento
# ──────────────────────────────────────────────

def train(draws: list[dict], transitions: dict, learning: dict,
          verbose_every: int = 0) -> dict:
    """
    Recorre los pares (draw[i] → draw[i+1]) y acumula el historial de
    predicciones. Retorna el learning actualizado.

    No recalcula los factores en cada ronda — lo hace UNA SOLA VEZ al final,
    lo que es mucho más eficiente y produce el mismo resultado.
    """
    history   = learning["history"]
    factors   = learning["factors"]
    biases    = learning["biases"]

    # Índice de fechas ya en el historial para no duplicar
    existing_dates = {
        (h["date"], h.get("sorteo", "?"))
        for h in history if h.get("actual")
    }

    new_rounds  = 0
    skipped     = 0
    total_hits  = [0] * NUM_GROUPS   # hits acumulados por grupo

    n = len(draws)
    print(f"\n  Procesando {n - 1} pares de entrenamiento...")

    for i in range(n - 1):
        current  = draws[i]
        next_day = draws[i + 1]
        key      = (current["date"].isoformat(), transitions["metadata"]["sorteo"])

        if key in existing_dates:
            skipped += 1
            continue

        # ── Predicción ────────────────────────────────────────────────────
        scores = compute_scores(current["numbers"], transitions, factors, biases)
        ranked = rank_candidates(scores)
        groups = build_groups(ranked)

        predicted_groups = {
            f"elite_{j + 1}": [num for num, _ in groups[f"elite_{j + 1}"]]
            for j in range(NUM_GROUPS)
        }

        # ── Evaluación vs resultado real ──────────────────────────────────
        actual_set = set(next_day["numbers"])

        hits_per_group  = {}
        hits_cumulative = {}
        cumulative      = set()
        for j in range(NUM_GROUPS):
            gkey = f"elite_{j + 1}"
            gset = set(predicted_groups[gkey])
            hits_per_group[gkey] = len(actual_set & gset)
            cumulative |= gset
            hits_cumulative[gkey] = len(actual_set & cumulative)
            total_hits[j] += hits_per_group[gkey]

        # ── Registrar en historial ────────────────────────────────────────
        entry = {
            "date":             current["date"].isoformat(),
            "sorteo":           transitions["metadata"]["sorteo"],
            "input":            current["numbers"],
            "predicted_groups": predicted_groups,
            "actual":           next_day["numbers"],
            "hits_per_group":   hits_per_group,
            "hits_cumulative":  hits_cumulative,
        }
        history.append(entry)
        existing_dates.add(key)
        new_rounds += 1

        # ── Verbose ──────────────────────────────────────────────────────
        if verbose_every > 0 and new_rounds % verbose_every == 0:
            acum = hits_cumulative.get(f"elite_{NUM_GROUPS}", 0)
            avg  = sum(total_hits) / new_rounds
            print(f"    [{new_rounds:>5}] {current['date']}  "
                  f"acum_hits={acum}/20  "
                  f"avg_hits/ronda={avg:.2f}")

    # ── Recalcular factores UNA VEZ con todo el historial ─────────────────
    print(f"\n  Recalculando factores de aprendizaje desde {len(history)} rondas...")
    new_factors, new_biases, new_stats = recompute_learning_from_history(history)
    learning["factors"] = new_factors
    learning["biases"]  = new_biases
    learning["stats"]   = new_stats
    learning["history"] = history

    return learning, new_rounds, skipped, total_hits


# ──────────────────────────────────────────────
# Resumen final
# ──────────────────────────────────────────────

def print_summary(sorteo: str, new_rounds: int, skipped: int,
                  total_hits: list[int], learning: dict):
    s_name = "Nocturna" if sorteo.upper() == "N" else "Vespertina"
    total  = sum(total_hits)
    completed = len([h for h in learning["history"] if h.get("actual")])

    print()
    print("=" * 60)
    print(f"  ENTRENAMIENTO COMPLETADO — {s_name} ({sorteo.upper()})")
    print("=" * 60)
    print(f"  Rondas nuevas procesadas : {new_rounds}")
    print(f"  Rondas ya existentes     : {skipped}")
    print(f"  Total en historial       : {completed}")
    print()
    print(f"  Aciertos por grupo (acumulado de {new_rounds} rondas nuevas):")
    print(f"  {'Grupo':<10} {'Hits':>6} {'Azar':>7} {'Mejora':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*8}")
    azar_unit = GROUP_SIZE / 100 * 20  # 6/100 * 20 = 1.2
    for j in range(NUM_GROUPS):
        gkey  = f"elite_{j+1}"
        h     = total_hits[j]
        azar  = round(azar_unit * new_rounds, 1)
        avg   = h / new_rounds if new_rounds else 0
        avg_a = azar_unit
        print(f"  {gkey:<10} {h:>6}  {azar:>6.1f}  {(avg - avg_a):>+7.3f}/ronda")

    # Top factores aprendidos
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
        print("  Top 5 números penalizados (lift < 1):")
        for k, lift, bias, pred, hits in low:
            prec = hits / pred if pred else 0
            print(f"    {k}  lift={lift:.3f}  bias={bias:+.3f}  "
                  f"precision={prec:.1%}  ({hits}/{pred})")

    print()
    print("  El modelo está listo. Próximo paso:")
    print(f"    python tombola_predict.py predict --sorteo {sorteo.upper()} \\")
    print(f"      --numbers <20 números del último sorteo>")
    print("=" * 60)
    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrenador automático del modelo de pronóstico — Tombola Uruguay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sorteo", default="N", choices=["N","V","n","v"],
                        help="Tipo de sorteo a entrenar (default: N=Nocturna)")
    parser.add_argument("--csv",
                        default=os.path.join(BASE_DIR, "tombolas.csv"),
                        help="Ruta al CSV (default: tombolas.csv junto al script)")
    parser.add_argument("--from", dest="date_from", default=None, metavar="YYYY-MM-DD",
                        help="Fecha de inicio del entrenamiento")
    parser.add_argument("--to", dest="date_to", default=None, metavar="YYYY-MM-DD",
                        help="Fecha de fin del entrenamiento")
    parser.add_argument("--last", type=int, default=None, metavar="N",
                        help="Entrenar solo los últimos N sorteos")
    parser.add_argument("--verbose", type=int, default=0, metavar="CADA_N",
                        help="Mostrar progreso cada N rondas (0 = silencioso)")
    parser.add_argument("--reset", action="store_true",
                        help="Borrar historial de aprendizaje antes de entrenar")
    args = parser.parse_args()

    sorteo = args.sorteo.upper()
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"

    print(f"\n  [tombola_train] Entrenando {s_name} ({sorteo})")
    print(f"  CSV: {args.csv}")

    # Cargar transiciones
    transitions = load_transitions(sorteo)

    # Cargar o resetear learning
    if args.reset:
        learning = {
            "factors": {f"{n:02d}": 1.0  for n in range(100)},
            "biases":  {f"{n:02d}": 0.0  for n in range(100)},
            "stats":   {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0}
                        for n in range(100)},
            "history": [],
        }
        print("  ⚠ Historial reseteado.")
    else:
        learning = load_learning(sorteo)
        existing = len([h for h in learning["history"] if h.get("actual")])
        print(f"  Historial existente: {existing} rondas")

    # Cargar sorteos del CSV
    draws = load_draws(args.csv, sorteo)
    print(f"  Sorteos en CSV: {len(draws)}  ({draws[0]['date']} → {draws[-1]['date']})")

    # Aplicar filtros de fecha
    if args.date_from:
        d_from = datetime.strptime(args.date_from, "%Y-%m-%d").date()
        draws = [d for d in draws if d["date"] >= d_from]
    if args.date_to:
        d_to = datetime.strptime(args.date_to, "%Y-%m-%d").date()
        draws = [d for d in draws if d["date"] <= d_to]
    if args.last:
        draws = draws[-(args.last + 1):]   # +1 porque el último no tiene siguiente

    if len(draws) < 2:
        print("ERROR: Se necesitan al menos 2 sorteos para entrenar.")
        sys.exit(1)

    print(f"  Rango a procesar: {draws[0]['date']} → {draws[-1]['date']}  "
          f"({len(draws) - 1} pares)")

    # Entrenar
    learning, new_rounds, skipped, total_hits = train(
        draws, transitions, learning, verbose_every=args.verbose
    )

    # Guardar
    save_learning(sorteo, learning)
    print(f"  ✓ Modelo guardado en tombola_{sorteo}_learning.json")

    # Resumen
    print_summary(sorteo, new_rounds, skipped, total_hits, learning)


if __name__ == "__main__":
    main()
