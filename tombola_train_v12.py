"""
tombola_train_v12.py  ·  Versión 1.2
──────────────────────────────────────
Entrena el modelo v1.2 usando los datos históricos de tombolas.csv.

Diferencias sobre tombola_train_v11.py:

  1. ENTRENAMIENTO DE 3 CAPAS:
     Por cada par (sorteo[i] → sorteo[i+1]) ejecuta la predicción en 3 capas
     (como lo hace tombola_predict_v12.py) y registra aciertos de cada capa.

  2. FACTORES POR CAPA:
     Al final del entrenamiento calcula factores/biases separados para:
       • factors      (Capa 1 — 24 números principales)
       • factors_l2   (Capa 2 — siguiente 24)
       • factors_l3   (Capa 3 — siguiente 24)
     Esto permite que cada capa mejore su precisión independientemente.

  3. SEÑALES MULTI-SALTO:
     Usa tombola_N_transitions_v12.json que incluye next_day2_all y next_day3_all.

USO
───
# Primera vez — reset completo con TODOS los resultados disponibles
python tombola_train_v12.py --sorteo N --reset --verbose 500

# Actualización incremental: agrega nuevos sorteos del CSV, recalcula factores
# desde el historial completo acumulado (no borra nada)
python tombola_train_v12.py --sorteo N --verbose 50

# Rango específico (solo si se quiere limitar por fecha)
python tombola_train_v12.py --sorteo N --from 2024-01-01 --to 2026-06-28

# Vespertina (igual, todos los resultados)
python tombola_train_v12.py --sorteo V --reset --verbose 500

REQUIERE
────────
tombola_N_transitions_v12.json  (generado por tombola_analysis_v12.py)
"""

import csv
import json
import math
import argparse
import os
import sys
import re
from collections import defaultdict
from datetime import date, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from tombola_predict_v12 import (
    load_transitions,
    load_learning,
    save_learning,
    compute_scores_v12,
    rank_candidates,
    build_groups,
    recompute_learning_v12,
    _extract_used_from_groups,
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
          verbose_every: int = 0) -> tuple[dict, int, int, list[int], list[int], list[int], list[int]]:
    """
    Recorre pares (draw[i] → draw[i+1]) y genera historial con 4 capas.

    Para cada ronda:
      • Capa 1: predicción con factors (globales / DOW)
      • Capa 2: predicción con factors_l2 sobre el pool restante
      • Capa 3: predicción con factors_l3 sobre el pool restante
      • Capa 4: predicción con factors_l4 sobre el pool restante

    Recalcula los factores de las 4 capas al final con decay temporal.
    """
    history  = learning["history"]
    factors  = learning["factors"]
    biases   = learning["biases"]
    f_l2     = learning.get("factors_l2", factors)
    b_l2     = learning.get("biases_l2",  biases)
    f_l3     = learning.get("factors_l3", factors)
    b_l3     = learning.get("biases_l3",  biases)
    f_l4     = learning.get("factors_l4", factors)
    b_l4     = learning.get("biases_l4",  biases)

    # Claves de rondas ya procesadas
    existing = {(h["date"], h.get("sorteo", "?"))
                for h in history if h.get("actual")}

    new_rounds = 0
    skipped    = 0
    total_hits_l1 = [0] * NUM_GROUPS
    total_hits_l2 = [0] * NUM_GROUPS
    total_hits_l3 = [0] * NUM_GROUPS
    total_hits_l4 = [0] * NUM_GROUPS

    n = len(draws)
    print(f"\n  Procesando {n - 1} pares (4 capas por ronda)...")

    for i in range(n - 1):
        cur = draws[i]
        nxt = draws[i + 1]
        key = (cur["date"].isoformat(), transitions["metadata"]["sorteo"])

        if key in existing:
            skipped += 1
            continue

        dow = cur["dow"]

        # ── Capa 1 — factores globales ──────────────────────────────────────
        scores1 = compute_scores_v12(cur["numbers"], transitions, factors, biases, dow)
        ranked1 = rank_candidates(scores1)
        hl_s1   = dict(learning.get("stats", {}))
        hl_s1["__factors__"] = factors
        groups1 = build_groups(ranked1, cur["numbers"], hl_s1)
        used1   = _extract_used_from_groups(groups1)

        # ── Capa 2 — factores propios l2 ────────────────────────────────────
        scores2 = compute_scores_v12(cur["numbers"], transitions, f_l2, b_l2, dow)
        ranked2 = sorted([(num, scores2[num]) for num in scores2 if num not in used1],
                         key=lambda x: x[1], reverse=True)
        hl_s2   = dict(learning.get("stats_l2", {}))
        hl_s2["__factors__"] = f_l2
        groups2 = build_groups(ranked2, cur["numbers"], hl_s2)
        used2   = used1 | _extract_used_from_groups(groups2)

        # ── Capa 3 — factores propios l3 ────────────────────────────────────
        scores3 = compute_scores_v12(cur["numbers"], transitions, f_l3, b_l3, dow)
        ranked3 = sorted([(num, scores3[num]) for num in scores3 if num not in used2],
                         key=lambda x: x[1], reverse=True)
        hl_s3   = dict(learning.get("stats_l3", {}))
        hl_s3["__factors__"] = f_l3
        groups3 = build_groups(ranked3, cur["numbers"], hl_s3)
        used3   = used2 | _extract_used_from_groups(groups3)

        # ── Capa 4 — factores propios l4 ────────────────────────────────────
        scores4 = compute_scores_v12(cur["numbers"], transitions, f_l4, b_l4, dow)
        ranked4 = sorted([(num, scores4[num]) for num in scores4 if num not in used3],
                         key=lambda x: x[1], reverse=True)
        hl_s4   = dict(learning.get("stats_l4", {}))
        hl_s4["__factors__"] = f_l4
        groups4 = build_groups(ranked4, cur["numbers"], hl_s4)

        # ── Evaluación ───────────────────────────────────────────────────────
        actual_set = set(nxt["numbers"])

        def _eval_hits(groups):
            hpg  = {}; hcum = {}; cumset: set = set()
            for j in range(NUM_GROUPS):
                gk = f"elite_{j+1}"
                gs = set(n for n, _ in groups.get(gk, []))
                hpg[gk]  = len(actual_set & gs)
                cumset  |= gs
                hcum[gk] = len(actual_set & cumset)
            return hpg, hcum

        def _pred(groups):
            return {f"elite_{j+1}": [n for n, _ in groups.get(f"elite_{j+1}", [])]
                    for j in range(NUM_GROUPS)}

        hpg1, hcum1 = _eval_hits(groups1)
        hpg2, hcum2 = _eval_hits(groups2)
        hpg3, hcum3 = _eval_hits(groups3)
        hpg4, hcum4 = _eval_hits(groups4)

        for j in range(NUM_GROUPS):
            gk = f"elite_{j+1}"
            total_hits_l1[j] += hpg1.get(gk, 0)
            total_hits_l2[j] += hpg2.get(gk, 0)
            total_hits_l3[j] += hpg3.get(gk, 0)
            total_hits_l4[j] += hpg4.get(gk, 0)

        history.append({
            "date":                cur["date"].isoformat(),
            "sorteo":              transitions["metadata"]["sorteo"],
            "dow":                 dow,
            "input":               cur["numbers"],
            "predicted_groups":    _pred(groups1),
            "predicted_groups_l2": _pred(groups2),
            "predicted_groups_l3": _pred(groups3),
            "predicted_groups_l4": _pred(groups4),
            "actual":              nxt["numbers"],
            "hits_per_group":      hpg1,
            "hits_cumulative":     hcum1,
            "hits_l2":             hpg2,
            "hits_cumulative_l2":  hcum2,
            "hits_l3":             hpg3,
            "hits_cumulative_l3":  hcum3,
            "hits_l4":             hpg4,
            "hits_cumulative_l4":  hcum4,
        })
        existing.add(key)
        new_rounds += 1

        if verbose_every > 0 and new_rounds % verbose_every == 0:
            acum1 = hcum1.get(f"elite_{NUM_GROUPS}", 0)
            acum2 = hcum2.get(f"elite_{NUM_GROUPS}", 0)
            acum3 = hcum3.get(f"elite_{NUM_GROUPS}", 0)
            acum4 = hcum4.get(f"elite_{NUM_GROUPS}", 0)
            print(f"    [{new_rounds:>5}]  {cur['date']}  DOW={dow}  "
                  f"C1={acum1}/20  C2={acum2}/20  C3={acum3}/20  C4={acum4}/20")

    # ── Recalcular factores de 4 capas ───────────────────────────────────────
    print(f"\n  Recalculando factores (DECAY={DECAY}) para 4 capas "
          f"desde {len(history)} rondas...")

    (new_f,  new_b,  new_s,
     new_f2, new_b2, new_s2,
     new_f3, new_b3, new_s3,
     new_f4, new_b4, new_s4,
     new_dof, new_dob) = recompute_learning_v12(history)

    learning.update({
        "factors":    new_f,   "biases":    new_b,   "stats":    new_s,
        "factors_l2": new_f2,  "biases_l2": new_b2,  "stats_l2": new_s2,
        "factors_l3": new_f3,  "biases_l3": new_b3,  "stats_l3": new_s3,
        "factors_l4": new_f4,  "biases_l4": new_b4,  "stats_l4": new_s4,
        "dow_factors": new_dof, "dow_biases": new_dob,
        "history":   history,
    })

    return learning, new_rounds, skipped, total_hits_l1, total_hits_l2, total_hits_l3, total_hits_l4


# ──────────────────────────────────────────────
# Resumen
# ──────────────────────────────────────────────

def print_summary(sorteo: str, new_rounds: int, skipped: int,
                  hits_l1: list, hits_l2: list, hits_l3: list, hits_l4: list,
                  learning: dict) -> None:
    s_name    = "Nocturna" if sorteo.upper() == "N" else "Vespertina"
    completed = len([h for h in learning["history"] if h.get("actual")])
    azar_unit = GROUP_SIZE / 100 * 20

    print()
    print("=" * 65)
    print(f"  ENTRENAMIENTO v1.2 COMPLETADO — {s_name} ({sorteo.upper()})")
    print("=" * 65)
    print(f"  Rondas nuevas procesadas  : {new_rounds}")
    print(f"  Rondas ya existentes      : {skipped}")
    print(f"  Total en historial        : {completed}")
    print(f"  Decay temporal            : {DECAY}  "
          f"(antiguo pesa {math.exp(-DECAY):.0%} del reciente)")
    print()

    for layer_num, hits, label in [
        (1, hits_l1, "Capa 1 PRINCIPAL   "),
        (2, hits_l2, "Capa 2 SECUNDARIA  "),
        (3, hits_l3, "Capa 3 TERCIARIA   "),
        (4, hits_l4, "Capa 4 CUATERNARIA "),
    ]:
        print(f"  Aciertos {label} ({new_rounds} rondas nuevas):")
        print(f"  {'Grupo':<10} {'Hits':>6} {'Azar':>7} {'Delta/ronda':>12}")
        print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*12}")
        for j in range(NUM_GROUPS):
            gk    = f"elite_{j+1}"
            h     = hits[j]
            azar  = round(azar_unit * new_rounds, 1)
            delta = h / max(1, new_rounds) - azar_unit
            print(f"  {gk:<10} {h:>6}  {azar:>6.1f}  {delta:>+11.3f}")
        print()

    # Top factores por capa
    for fkey, skey, bkey, layer_name in [
        ("factors",    "stats",    "biases",    "Capa 1"),
        ("factors_l2", "stats_l2", "biases_l2", "Capa 2"),
        ("factors_l3", "stats_l3", "biases_l3", "Capa 3"),
        ("factors_l4", "stats_l4", "biases_l4", "Capa 4"),
    ]:
        facts = learning.get(fkey, {}); st = learning.get(skey, {})
        bias_ = learning.get(bkey, {})
        reliable = [
            (k, facts[k], bias_.get(k, 0),
             st.get(k, {}).get("predicted", 0),
             st.get(k, {}).get("hits", 0))
            for k in facts
            if st.get(k, {}).get("predicted", 0) >= 10
        ]
        if not reliable:
            continue
        top = sorted(reliable, key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 {layer_name} — mayor lift:")
        for k, lift, bias, pred, hits in top:
            prec = hits / pred if pred else 0
            print(f"    {k}  lift={lift:.3f}  bias={bias:+.3f}  "
                  f"prec={prec:.1%}  ({hits}/{pred})")
        print()

    print("  Próximos pasos:")
    print(f"    # Actualización incremental (después de agregar sorteos al CSV):")
    print(f"    python tombola_train_v12.py --sorteo {sorteo.upper()} --verbose 50")
    print(f"")
    print(f"    # Predicción:")
    print(f"    python tombola_predict_v12.py predict --sorteo {sorteo.upper()} \\")
    print(f"      --date YYYY-MM-DD  --numbers <20 números>")
    print("=" * 65)
    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrenador v1.2 — Tombola Uruguay (3 capas + decay temporal)",
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
                        help="Borrar historial y factores antes de entrenar")
    args = parser.parse_args()

    sorteo = args.sorteo.upper()
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"

    print(f"\n  [tombola_train_v12] Entrenando {s_name} ({sorteo})")
    print(f"  CSV: {args.csv}")

    transitions = load_transitions(sorteo)

    _df = {f"{n:02d}": 1.0 for n in range(100)}
    _db = {f"{n:02d}": 0.0 for n in range(100)}
    _ds = {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0,
                        "w_predicted": 0.0, "w_hits": 0.0} for n in range(100)}

    if args.reset:
        import copy
        learning = {
            "factors":     copy.deepcopy(_df),  "biases":    copy.deepcopy(_db),
            "stats":       copy.deepcopy(_ds),
            "factors_l2":  copy.deepcopy(_df),  "biases_l2": copy.deepcopy(_db),
            "stats_l2":    copy.deepcopy(_ds),
            "factors_l3":  copy.deepcopy(_df),  "biases_l3": copy.deepcopy(_db),
            "stats_l3":    copy.deepcopy(_ds),
            "factors_l4":  copy.deepcopy(_df),  "biases_l4": copy.deepcopy(_db),
            "stats_l4":    copy.deepcopy(_ds),
            "dow_factors": {str(d): {f"{n:02d}": 1.0 for n in range(100)} for d in range(7)},
            "dow_biases":  {str(d): {f"{n:02d}": 0.0 for n in range(100)} for d in range(7)},
            "history":     [],
        }
        print("  ⚠ Historial y factores reseteados (4 capas).")
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

    learning, new_rounds, skipped, hits_l1, hits_l2, hits_l3, hits_l4 = train(
        draws, transitions, learning, verbose_every=args.verbose
    )

    save_learning(sorteo, learning)
    print(f"  ✓ Modelo guardado en tombola_{sorteo}_learning_v12.json")

    print_summary(sorteo, new_rounds, skipped, hits_l1, hits_l2, hits_l3, hits_l4, learning)


if __name__ == "__main__":
    main()
