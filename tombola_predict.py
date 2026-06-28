"""
tombola_predict.py
──────────────────
Sistema de pronóstico y aprendizaje para la Tombola uruguaya.

COMANDOS
────────
1. Predecir próximo sorteo (ingresar los 20 del día anterior):
   python tombola_predict.py predict --sorteo N --numbers 04 08 11 18 30 48 49 50 52 54 60 65 67 68 72 74 80 87 90 94

2. Registrar el resultado real (para que el sistema aprenda):
   python tombola_predict.py feedback --sorteo N --date 2026-05-23 --actual 00 06 07 08 10 16 17 19 27 28 31 40 46 49 56 61 74 80 90 98

3. Ver estadísticas de precisión del sistema:
   python tombola_predict.py accuracy --sorteo N

4. Ver historial de predicciones:
   python tombola_predict.py history --sorteo N --last 5

CÓMO FUNCIONA EL SCORING (v2)
──────────────────────────────
Paso 1 — Score base: para cada número de entrada se acumulan las probabilidades
  de transición del JSON. Se blendean datos históricos (60%) y recientes (40%).

Paso 2 — Normalización por rango: los 100 scores brutos se convierten a un
  score de rango en [0, 1] (el #1 = 1.0, el #100 = 0.0). Esto amplifica las
  diferencias reales, que en valores brutos son muy pequeñas (~0.02 de rango).

Paso 3 — Factor de lift + bias aprendido:
  final_score(C) = rank_score(C) × lift(C) + bias(C)

  lift(C) y bias(C) se recalculan desde cero con cada `feedback` usando todo
  el historial acumulado (no EMA). Esto da el aprendizaje más rápido posible.

  lift(C) = precision_suavizada(C) / tasa_base_esperada
    - precision(C): cuántas veces C apareció de las veces que fue predicho
    - tasa_base: 20/100 = 0.20 (probabilidad aleatoria de cualquier número)
    - Suavizado de Laplace para estabilidad con pocos datos

  bias(C) = amplificación de (precision - tasa_base) → mueve el ranking
    incluso cuando el lift aún es débil

GRUPOS DE SALIDA
────────────────
  5 grupos de 6 números (ÉLITE 1 → ÉLITE 5), cubre el top 30 (30% del espacio)
  Con el modelo puro (sin aprendizaje): ~6 hits esperados sobre 20 números reales
  Con aprendizaje activo: el sistema aprende qué números el modelo subestima/sobreestima
"""

import json
import argparse
import os
import sys
from datetime import date, datetime
from collections import defaultdict


# ──────────────────────────────────────────────
# Rutas por defecto
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def transitions_path(sorteo: str) -> str:
    return os.path.join(BASE_DIR, f"tombola_{sorteo.upper()}_transitions.json")

def learning_path(sorteo: str) -> str:
    return os.path.join(BASE_DIR, f"tombola_{sorteo.upper()}_learning.json")


# ──────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────

def load_transitions(sorteo: str) -> dict:
    path = transitions_path(sorteo)
    if not os.path.exists(path):
        print(f"ERROR: No se encontró {path}")
        print(f"Ejecuta primero: python tombola_analysis.py --sorteo {sorteo}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_learning(sorteo: str) -> dict:
    """
    Estructura del archivo de aprendizaje:
    {
      "factors": { "00": 1.0, ... },   ← lift por número (recalculado cada feedback)
      "biases":  { "00": 0.0, ... },   ← sesgo aditivo por número (recalculado)
      "stats":   {                      ← contadores crudos para diagnóstico
        "00": {"predicted": 0, "appeared": 0, "hits": 0},
        ...
      },
      "history": [...]
    }
    """
    path = learning_path(sorteo)
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        # migración: si falta biases/stats, añadirlos
        if "biases" not in data:
            data["biases"] = {f"{n:02d}": 0.0 for n in range(100)}
        if "stats" not in data:
            data["stats"] = {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0}
                             for n in range(100)}
        return data
    return {
        "factors": {f"{n:02d}": 1.0  for n in range(100)},
        "biases":  {f"{n:02d}": 0.0  for n in range(100)},
        "stats":   {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0}
                    for n in range(100)},
        "history": [],
    }


def save_learning(sorteo: str, data: dict):
    path = learning_path(sorteo)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Núcleo del scoring
# ──────────────────────────────────────────────

def compute_scores(input_numbers: list[str], transitions: dict,
                   factors: dict, biases: dict) -> dict[str, float]:
    """
    Calcula el score final para cada candidato (00-99).

    Pipeline:
      1. Acumula probabilidades de transición (60% históricas + 40% recientes).
      2. Convierte a score de RANGO [0,1] → amplifica diferencias reales pequeñas.
      3. Aplica lift (factor multiplicativo) + bias aditivo del aprendizaje.

    Retorna dict { "00": 0.87, "01": 0.54, ... }  (escala arbitraria, solo importa el orden)
    """
    raw = defaultdict(float)

    for num in input_numbers:
        num = f"{int(num):02d}"
        entry = transitions["transitions"].get(num)
        if not entry:
            continue
        hist   = entry.get("next_day", [])
        recent = entry.get("recent_next_day", [])
        w_hist   = 0.6 if recent else 1.0
        w_recent = 0.4 if recent else 0.0
        for c, pct, _ in hist:
            raw[c] += pct * w_hist
        for c, pct, _ in recent:
            raw[c] += pct * w_recent

    # Normalizar por cantidad de entradas
    n = len(input_numbers)
    base = {f"{i:02d}": raw.get(f"{i:02d}", 0.0) / n for i in range(100)}

    # ── Normalización por rango [0, 1] ──────────────────────────────────────
    # Convierte el ranking de scores brutos a una escala uniforme.
    # Así, el #1 siempre vale 1.0 y el #100 vale 0.0, sin importar cuán
    # apretados estén los valores brutos (problema: rango bruto ≈ 0.02).
    sorted_nums = sorted(base, key=base.get, reverse=True)
    rank_score  = {num: 1.0 - (i / 99.0) for i, num in enumerate(sorted_nums)}

    # ── Aplicar lift × bias del aprendizaje ─────────────────────────────────
    final = {}
    for k, rs in rank_score.items():
        f = factors.get(k, 1.0)
        b = biases.get(k, 0.0)
        final[k] = round(max(0.0, rs * f + b), 5)

    return final


def rank_candidates(scores: dict[str, float]) -> list[tuple[str, float]]:
    """Retorna lista [(num, score), ...] ordenada de mayor a menor."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


GROUP_SIZE = 6       # números por grupo élite
NUM_GROUPS = 5       # cantidad de grupos élite (cubre top 30)

def build_groups(ranked: list[tuple[str, float]]) -> dict:
    """
    Divide los 100 números en NUM_GROUPS grupos de GROUP_SIZE + resto.
    Ejemplo con 5 grupos de 6:
      elite_1: rank  1-6  (máxima confianza)
      elite_2: rank  7-12
      elite_3: rank 13-18
      elite_4: rank 19-24
      elite_5: rank 25-30
      resto:   rank 31-100
    """
    groups = {}
    for i in range(NUM_GROUPS):
        key = f"elite_{i + 1}"
        groups[key] = ranked[i * GROUP_SIZE : (i + 1) * GROUP_SIZE]
    covered = NUM_GROUPS * GROUP_SIZE
    groups["resto"] = ranked[covered:]
    return groups


# ──────────────────────────────────────────────
# Aprendizaje: lift + bias recalculados desde historial completo
# ──────────────────────────────────────────────

MAX_FACTOR  = 2.5
MIN_FACTOR  = 0.2
MAX_BIAS    = 0.35
MIN_BIAS    = -0.35
BASE_RATE   = 20 / 100   # prob. aleatoria de que cualquier número aparezca
LAPLACE_K   = 5          # pseudo-observaciones para suavizar con datos escasos
BIAS_SCALE  = 1.5        # amplificación del bias aditivo


def recompute_learning_from_history(history: list[dict]) -> tuple[dict, dict, dict]:
    """
    Recalcula DESDE CERO los factores de lift, biases y estadísticas usando
    todo el historial completado. No usa EMA — cada feedback regenera el modelo
    completo, lo que garantiza el aprendizaje más rápido y consistente posible.

    Retorna (factors, biases, stats).

    ─ Lift (factor multiplicativo) ────────────────────────────────────────────
      Para cada número n:
        predicted(n) = veces que n apareció en algún grupo élite predicho
        hits(n)      = de esas veces, cuántas n apareció en el resultado real
        precision_suavizada(n) = (hits + K × BASE_RATE) / (predicted + K)
        lift(n) = precision_suavizada / BASE_RATE

      lift > 1 → el modelo acierta más que el azar para n → se amplifica
      lift < 1 → el modelo falla más que el azar para n → se penaliza

    ─ Bias (término aditivo al rank_score) ────────────────────────────────────
      bias(n) = (precision_suavizada - BASE_RATE) × BIAS_SCALE
      Mueve el ranking incluso cuando el lift es débil (pocas rondas).
    """
    completed = [h for h in history
                 if h.get("actual") and h.get("predicted_groups")]
    total = len(completed)

    stats = {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0}
             for n in range(100)}

    for entry in completed:
        actual_set = set(entry["actual"])
        for num in actual_set:
            stats[num]["appeared"] += 1
        for nums in entry.get("predicted_groups", {}).values():
            for num in nums:
                stats[num]["predicted"] += 1
                if num in actual_set:
                    stats[num]["hits"] += 1

    factors = {}
    biases  = {}

    for n in range(100):
        k   = f"{n:02d}"
        p   = stats[k]["predicted"]
        h   = stats[k]["hits"]

        # Laplace smoothing
        precision = (h + LAPLACE_K * BASE_RATE) / (p + LAPLACE_K)

        # Lift
        lift = precision / BASE_RATE
        factors[k] = round(max(MIN_FACTOR, min(MAX_FACTOR, lift)), 4)

        # Bias
        raw_bias = (precision - BASE_RATE) * BIAS_SCALE
        biases[k] = round(max(MIN_BIAS, min(MAX_BIAS, raw_bias)), 4)

    return factors, biases, stats


# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────

def fmt_group(group_items: list[tuple[str, float]], cols: int = 5) -> str:
    lines = []
    row = []
    for i, (num, score) in enumerate(group_items):
        row.append(f"{num}({score:.3f})")
        if (i + 1) % cols == 0:
            lines.append("  " + "  ".join(row))
            row = []
    if row:
        lines.append("  " + "  ".join(row))
    return "\n".join(lines)


def fmt_group_plain(group_items: list[tuple[str, float]], cols: int = 10) -> str:
    nums = [num for num, _ in group_items]
    lines = []
    for i in range(0, len(nums), cols):
        lines.append("  " + "  ".join(nums[i:i+cols]))
    return "\n".join(lines)


GROUP_LABELS = {
    "elite_1": ("🥇", "ÉLITE 1", "máxima confianza"),
    "elite_2": ("🥈", "ÉLITE 2", "alta confianza"),
    "elite_3": ("🥉", "ÉLITE 3", "buena confianza"),
    "elite_4": ("🔵", "ÉLITE 4", "confianza media-alta"),
    "elite_5": ("🟡", "ÉLITE 5", "confianza media"),
}


def print_prediction(sorteo: str, input_numbers: list[str], groups: dict,
                     learning_data: dict, show_scores: bool = True):
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"
    total_rounds = len(learning_data["history"])

    print()
    print("=" * 56)
    print(f"  PRONÓSTICO TOMBOLA {s_name.upper()} ({sorteo})")
    print("=" * 56)
    print(f"  Entrada ({len(input_numbers)} números): {' '.join(sorted(input_numbers))}")
    print(f"  Rondas aprendidas: {total_rounds}")
    print()

    for i in range(NUM_GROUPS):
        key = f"elite_{i + 1}"
        items = groups.get(key, [])
        emoji, label, desc = GROUP_LABELS.get(key, ("⚪", key.upper(), ""))
        header = f"{emoji} {label}  ({GROUP_SIZE} números — {desc})"
        print(f"  {header}")
        if show_scores:
            print(fmt_group(items, cols=6))
        else:
            print(fmt_group_plain(items, cols=6))
        print()

    # Resto en forma compacta
    resto_nums = " ".join(n for n, _ in groups.get("resto", []))
    covered = NUM_GROUPS * GROUP_SIZE
    print(f"  ⚪ RESTO     ({100 - covered} números — referencia)")
    print(f"  {resto_nums}")
    print()
    print("  Para registrar el resultado real:")
    print(f'  python tombola_predict.py feedback --sorteo {sorteo} \\')
    print(f'    --date {date.today().isoformat()} \\')
    print(f'    --actual <20 numeros que salieron>')
    print("=" * 56)
    print()


# ──────────────────────────────────────────────
# Comando: predict
# ──────────────────────────────────────────────

def cmd_predict(args):
    sorteo = args.sorteo.upper()
    transitions = load_transitions(sorteo)
    learning = load_learning(sorteo)

    # Validar y normalizar números de entrada
    raw = args.numbers
    if len(raw) != 20:
        print(f"ERROR: Se necesitan exactamente 20 números, se recibieron {len(raw)}.")
        sys.exit(1)
    try:
        input_numbers = [f"{int(n):02d}" for n in raw]
    except ValueError:
        print("ERROR: Todos los números deben ser enteros entre 00 y 99.")
        sys.exit(1)

    scores = compute_scores(input_numbers, transitions,
                            learning["factors"], learning["biases"])
    ranked = rank_candidates(scores)
    groups = build_groups(ranked)

    print_prediction(sorteo, input_numbers, groups, learning,
                     show_scores=not args.plain)

    # Guardar la predicción pendiente para cuando llegue el feedback
    predicted_groups = {
        f"elite_{i + 1}": [n for n, _ in groups.get(f"elite_{i + 1}", [])]
        for i in range(NUM_GROUPS)
    }
    pending = {
        "date": args.date or date.today().isoformat(),
        "sorteo": sorteo,
        "input": input_numbers,
        "predicted_groups": predicted_groups,   # { "elite_1": [...], ..., "elite_5": [...] }
        "actual": None,
        "hits_per_group": None,                 # { "elite_1": n, ..., "elite_5": n }
        "hits_cumulative": None,                # hits acumulados elite1, elite1+2, ...
    }

    # Agregar o actualizar entrada pendiente para esta fecha
    history = learning["history"]
    existing = next((i for i, h in enumerate(history)
                     if h["date"] == pending["date"] and h["sorteo"] == sorteo
                     and h["actual"] is None), None)
    if existing is not None:
        history[existing] = pending
    else:
        history.append(pending)

    save_learning(sorteo, learning)
    print(f"  ✓ Predicción guardada. Registra el resultado con `feedback` para entrenar el sistema.")
    print()


# ──────────────────────────────────────────────
# Comando: feedback
# ──────────────────────────────────────────────

def cmd_feedback(args):
    sorteo = args.sorteo.upper()
    transitions = load_transitions(sorteo)
    learning = load_learning(sorteo)

    try:
        actual = [f"{int(n):02d}" for n in args.actual]
    except ValueError:
        print("ERROR: Números inválidos en --actual.")
        sys.exit(1)

    if len(actual) != 20:
        print(f"ERROR: Se necesitan 20 números reales, se recibieron {len(actual)}.")
        sys.exit(1)

    target_date = args.date or date.today().isoformat()

    # Buscar predicción pendiente para esa fecha
    history = learning["history"]
    entry_idx = next(
        (i for i, h in enumerate(history)
         if h["date"] == target_date and h["sorteo"] == sorteo and h["actual"] is None),
        None
    )

    if entry_idx is None:
        print(f"AVISO: No se encontró predicción pendiente para {target_date} ({sorteo}).")
        print("Registrando resultado sin predicción previa (solo para historial).")
        history.append({
            "date": target_date, "sorteo": sorteo,
            "input": [], "predicted_groups": {}, "actual": actual,
            "hits_per_group": None, "hits_cumulative": None,
        })
        save_learning(sorteo, learning)
        return

    entry = history[entry_idx]
    actual_set = set(actual)
    predicted_groups = entry.get("predicted_groups", {})

    # Hits por grupo y acumulados
    hits_per_group = {}
    cumulative_set: set[str] = set()
    hits_cumulative = {}
    for i in range(NUM_GROUPS):
        key = f"elite_{i + 1}"
        group_set = set(predicted_groups.get(key, []))
        hits_per_group[key] = len(actual_set & group_set)
        cumulative_set |= group_set
        hits_cumulative[key] = len(actual_set & cumulative_set)

    entry["actual"]          = actual
    entry["hits_per_group"]  = hits_per_group
    entry["hits_cumulative"] = hits_cumulative

    # Recalcular lift + bias desde todo el historial (máximo aprendizaje)
    new_factors, new_biases, new_stats = recompute_learning_from_history(history)
    learning["factors"] = new_factors
    learning["biases"]  = new_biases
    learning["stats"]   = new_stats
    history[entry_idx]  = entry
    save_learning(sorteo, learning)

    s_name = "Nocturna" if sorteo == "N" else "Vespertina"
    print()
    print("=" * 54)
    print(f"  RESULTADO REGISTRADO — {s_name} {target_date}")
    print("=" * 54)
    print(f"  Números reales: {' '.join(sorted(actual))}")
    print()
    print(f"  {'Grupo':<10} {'Predichos':>9} {'Aciertos':>9} {'Acum.':>7} {'Azar':>6}")
    print(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*7} {'-'*6}")
    acum_count = 0
    for i in range(NUM_GROUPS):
        key = f"elite_{i + 1}"
        h   = hits_per_group[key]
        hc  = hits_cumulative[key]
        acum_count += GROUP_SIZE
        azar = round(acum_count / 100 * 20, 1)
        print(f"  {key:<10} {GROUP_SIZE:>9} {h:>7}/20  {hc:>5}/20  {azar:>5}")
    print()

    # Números no capturados
    all_predicted = set(n for g in predicted_groups.values() for n in g)
    missed = sorted(actual_set - all_predicted)
    if missed:
        print(f"  No predichos en ningún grupo: {' '.join(missed)}")
    print()
    print(f"  ✓ Factores de aprendizaje actualizados.")
    print("=" * 54)
    print()


# ──────────────────────────────────────────────
# Comando: accuracy
# ──────────────────────────────────────────────

def cmd_accuracy(args):
    sorteo = args.sorteo.upper()
    learning = load_learning(sorteo)
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"

    completed = [h for h in learning["history"]
                 if h.get("actual") and h.get("hits_per_group") is not None]

    if not completed:
        print(f"Sin rondas completadas todavía para {s_name}.")
        return

    n = len(completed)
    print()
    print("=" * 58)
    print(f"  PRECISIÓN DEL SISTEMA — {s_name} ({sorteo})")
    print("=" * 58)
    print(f"  Rondas evaluadas: {n}")
    print()
    print(f"  {'Grupo':<10} {'Nums':>5} {'Prom.aciertos':>14} {'Azar':>6} {'Mejora':>8}")
    print(f"  {'-'*10} {'-'*5} {'-'*14} {'-'*6} {'-'*8}")

    acum_count = 0
    for i in range(NUM_GROUPS):
        key = f"elite_{i + 1}"
        acum_count += GROUP_SIZE
        # Hits acumulados hasta este grupo
        avg_cum = sum(h["hits_cumulative"].get(key, 0) for h in completed if h.get("hits_cumulative")) / n
        # Hits solo de este grupo
        avg_solo = sum(h["hits_per_group"].get(key, 0) for h in completed if h.get("hits_per_group")) / n
        azar_solo = round(GROUP_SIZE / 100 * 20, 2)
        azar_cum  = round(acum_count / 100 * 20, 2)
        print(f"  {key:<10} {GROUP_SIZE:>5} {avg_solo:>8.2f}/20  (acum {avg_cum:.2f})   azar solo:{azar_solo:.1f}")

    print()
    print("  (azar = esperado si se eligieran al azar)")
    print()

    # Últimas 5 rondas con hits por grupo
    last = completed[-5:]
    header_cols = "  ".join(f"E{i+1}" for i in range(NUM_GROUPS))
    print(f"  Últimas {len(last)} rondas  (hits por grupo de {GROUP_SIZE}):")
    print(f"  {'Fecha':<12}  {header_cols}  | acum E1+…+E5")
    for h in last:
        hpg = h.get("hits_per_group", {})
        hcm = h.get("hits_cumulative", {})
        cols = "   ".join(str(hpg.get(f"elite_{i+1}", "-")) for i in range(NUM_GROUPS))
        total = hcm.get(f"elite_{NUM_GROUPS}", "-")
        print(f"  {h['date']:<12}  {cols}  |   {total}/20")

    # Números con mayor lift aprendido (los que el modelo identifica mejor)
    stats = learning.get("stats", {})
    # Solo mostrar números con al menos 3 predicciones (datos suficientes)
    reliable = [(k, learning["factors"][k], learning["biases"].get(k, 0),
                 stats.get(k, {}).get("predicted", 0),
                 stats.get(k, {}).get("hits", 0))
                for k in learning["factors"]
                if stats.get(k, {}).get("predicted", 0) >= 3]

    if reliable:
        top_lift = sorted(reliable, key=lambda x: x[1], reverse=True)[:8]
        low_lift = sorted(reliable, key=lambda x: x[1])[:5]
        print()
        print(f"  Números mejor aprendidos (lift ≥ 1, min 3 predicciones):")
        for k, lift, bias, pred, hits in top_lift:
            prec = hits / pred if pred else 0
            print(f"    {k}  lift={lift:.2f}  bias={bias:+.3f}  "
                  f"hits={hits}/{pred}  precision={prec:.0%}  (base={BASE_RATE:.0%})")
        print()
        print(f"  Números penalizados (lift < 1):")
        for k, lift, bias, pred, hits in low_lift:
            prec = hits / pred if pred else 0
            print(f"    {k}  lift={lift:.2f}  bias={bias:+.3f}  "
                  f"hits={hits}/{pred}  precision={prec:.0%}")
    else:
        print()
        print("  (Se necesitan al menos 3 predicciones por número para mostrar lift confiable)")
    print("=" * 58)
    print()


# ──────────────────────────────────────────────
# Comando: history
# ──────────────────────────────────────────────

def cmd_history(args):
    sorteo = args.sorteo.upper()
    learning = load_learning(sorteo)
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"

    history = learning["history"]
    if args.last:
        history = history[-args.last:]

    if not history:
        print(f"Sin historial para {s_name}.")
        return

    header_cols = "  ".join(f"E{i+1}" for i in range(NUM_GROUPS))
    print()
    print(f"  HISTORIAL {s_name} — últimas {len(history)} entradas")
    print(f"  {'Fecha':<12}  {header_cols}  | total  Estado")
    print(f"  {'-'*12}  {'  '.join(['-'*2]*NUM_GROUPS)}  | -----  --------")
    for h in history:
        if h.get("actual") and h.get("hits_per_group"):
            hpg   = h["hits_per_group"]
            hcm   = h.get("hits_cumulative", {})
            cols  = "   ".join(str(hpg.get(f"elite_{i+1}", "-")) for i in range(NUM_GROUPS))
            total = hcm.get(f"elite_{NUM_GROUPS}", "?")
            estado = f"{total}/20  ✓"
        else:
            cols  = "  ".join([" -"] * NUM_GROUPS)
            estado = "⏳ pendiente"
        print(f"  {h['date']:<12}  {cols}  |  {estado}")
    print()


# ──────────────────────────────────────────────
# CLI principal
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de pronóstico y aprendizaje — Tombola Uruguay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    p_pred = sub.add_parser("predict", help="Generar pronóstico para el próximo sorteo")
    p_pred.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    p_pred.add_argument("--numbers", nargs=20, required=True, metavar="NUM",
                        help="Los 20 números del sorteo anterior")
    p_pred.add_argument("--date", default=None,
                        help="Fecha del sorteo a predecir (YYYY-MM-DD, default: hoy)")
    p_pred.add_argument("--plain", action="store_true",
                        help="Mostrar números sin scores (más limpio)")

    # feedback
    p_fb = sub.add_parser("feedback", help="Registrar resultado real para aprendizaje")
    p_fb.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    p_fb.add_argument("--actual", nargs=20, required=True, metavar="NUM",
                      help="Los 20 números que salieron realmente")
    p_fb.add_argument("--date", default=None,
                      help="Fecha del sorteo (YYYY-MM-DD, default: hoy)")

    # accuracy
    p_acc = sub.add_parser("accuracy", help="Ver estadísticas de precisión")
    p_acc.add_argument("--sorteo", default="N", choices=["N","V","n","v"])

    # history
    p_hist = sub.add_parser("history", help="Ver historial de predicciones")
    p_hist.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    p_hist.add_argument("--last", type=int, default=10, metavar="N",
                        help="Mostrar las últimas N entradas (default: 10)")

    args = parser.parse_args()

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "feedback":
        cmd_feedback(args)
    elif args.command == "accuracy":
        cmd_accuracy(args)
    elif args.command == "history":
        cmd_history(args)


if __name__ == "__main__":
    main()
