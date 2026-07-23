"""
tombola_predict_v12.py  ·  Versión 1.2
────────────────────────────────────────
Mejoras sobre v1.1:

1. PREDICCIÓN EN 4 CAPAS:
   • Capa 1 (PRINCIPAL):    top 24 con factores aprendidos de Capa 1.
   • Capa 2 (SECUNDARIA):   76 restantes → top 24 con factores propios de Capa 2.
   • Capa 3 (TERCIARIA):    52 restantes → top 24 con factores propios de Capa 3.
   • Capa 4 (CUATERNARIA):  28 restantes → top 24 con factores propios de Capa 4.
   • Resto final:            4 números de menor probabilidad.

2. FACTORES POR CAPA:
   Cada capa aprende sus propios lift/bias a partir del entrenamiento.
   Los números que habitualmente caen en cada capa reciben boosts específicos,
   mejorando la cobertura del resultado real.

3. SEÑALES MULTI-SALTO (requiere tombola_analysis_v12.py):
   next_day2_all y next_day3_all capturan patrones con brecha de 2 y 3 sorteos.

Archivos propios (generados por tombola_analysis_v12 y tombola_train_v12):
  tombola_N_transitions_v12.json  (análisis con multi-salto)
  tombola_N_learning_v12.json     (factores por capa)

COMANDOS
────────
1. Generar análisis (una vez, luego de cada actualización del CSV):
   python tombola_analysis_v12.py --sorteo N

2. Entrenar:
   python tombola_train_v12.py --sorteo N --reset --last 500 --verbose 100

3. Predecir:
   python tombola_predict_v12.py predict --sorteo N --date 2026-06-27 \\
     --numbers 09 11 14 23 25 27 31 49 50 58 59 61 62 63 68 75 78 84 93 97

4. Registrar resultado:
   python tombola_predict_v12.py feedback --sorteo N --date 2026-06-26 \\
     --actual 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19

5. Estadísticas:
   python tombola_predict_v12.py accuracy --sorteo N

6. Historial:
   python tombola_predict_v12.py history --sorteo N --last 5
"""

import json
import math
import argparse
import os
import sys
from datetime import date, datetime
from collections import defaultdict
from itertools import combinations


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Pesos de ventana v1.2 ─────────────────────────────────────────────────────
# Los pesos multi-salto reducen proporcionalmente el resto respecto a v1.1.
WINDOW_WEIGHTS = {
    "next_day_5s":    0.07,
    "next_day_30d":   0.08,
    "next_day_90d":   0.19,
    "next_day_365d":  0.27,
    "next_day_all":   0.21,
    "next_day2_all":  0.12,   # brecha 2 sorteos — señal propia de v1.2
    "next_day3_all":  0.06,   # brecha 3 sorteos — señal propia de v1.2
}

# ── Ajuste DOW ────────────────────────────────────────────────────────────────
DOW_INFLUENCE = 0.12

# ── Señal Cold/Carryover/Skip-day ─────────────────────────────────────────────
COLD_THRESHOLD   = 25
COLD_BOOST       = 0.010
CARRYOVER_BOOST  = 0.020
SKIPDAY_BOOST    = 0.010

# ── Concentración de decil en input ───────────────────────────────────────────
INPUT_DECIL_CONC_THR       = 4
INPUT_DECIL_CONC_INFLUENCE = 0.06

# ── Grupos de apuesta ─────────────────────────────────────────────────────────
MAX_PER_DECIL       = 2
MIN_CARRYOVER_SLOTS = 3
MIN_HIGHLIFT_SLOTS  = 2
HIGHLIFT_THR        = 1.25
HIGHLIFT_MIN_PRED   = 20
NUM_BETTING_GROUPS  = 2
SCORE_THRESHOLD     = 1.05
SCORE_HIGH          = 1.50

# ── Learning ──────────────────────────────────────────────────────────────────
GROUP_SIZE  = 6
NUM_GROUPS  = 4
MAX_FACTOR  = 2.5
MIN_FACTOR  = 0.2
MAX_BIAS    = 0.35
MIN_BIAS    = -0.35
BASE_RATE   = 20 / 100
LAPLACE_K   = 5
BIAS_SCALE  = 1.5
DECAY       = 2.0
MIN_DOW_ROUNDS = 30

# ── Ventana 5s adaptativa ─────────────────────────────────────────────────────
ADAPTIVE_TOP_N     = 15
ADAPTIVE_THRESHOLD = 0.40

# ── Co-ocurrencia ─────────────────────────────────────────────────────────────
COOCCUR_TOP_N      = 16
COOCCUR_INFLUENCE  = 0.30
NUM_COOCCUR_GROUPS = 3
COOCCUR_GROUP_SIZE = 4


# ──────────────────────────────────────────────
# Rutas — archivos propios de v1.2
# ──────────────────────────────────────────────

def transitions_path(sorteo: str) -> str:
    return os.path.join(BASE_DIR, f"tombola_{sorteo.upper()}_transitions_v12.json")

def learning_path(sorteo: str) -> str:
    return os.path.join(BASE_DIR, f"tombola_{sorteo.upper()}_learning_v12.json")


# ──────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────

def load_transitions(sorteo: str) -> dict:
    path = transitions_path(sorteo)
    if not os.path.exists(path):
        print(f"ERROR: No se encontró {path}")
        print(f"Ejecuta primero: python tombola_analysis_v12.py --sorteo {sorteo}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_learning(sorteo: str) -> dict:
    _df = {f"{n:02d}": 1.0 for n in range(100)}
    _db = {f"{n:02d}": 0.0 for n in range(100)}
    _ds = {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0,
                        "w_predicted": 0.0, "w_hits": 0.0} for n in range(100)}
    _ddowf = {str(d): {f"{n:02d}": 1.0 for n in range(100)} for d in range(7)}
    _ddowb = {str(d): {f"{n:02d}": 0.0 for n in range(100)} for d in range(7)}

    path = learning_path(sorteo)
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        for field, default in [
            ("factors",     _df), ("biases",  _db), ("stats",  _ds),
            ("factors_l2",  _df), ("biases_l2", _db), ("stats_l2", _ds),
            ("factors_l3",  _df), ("biases_l3", _db), ("stats_l3", _ds),
            ("factors_l4",  _df), ("biases_l4", _db), ("stats_l4", _ds),
            ("dow_factors", _ddowf), ("dow_biases", _ddowb),
        ]:
            if field not in data:
                import copy; data[field] = copy.deepcopy(default)
        return data

    import copy
    return {
        "factors":     copy.deepcopy(_df),  "biases":    copy.deepcopy(_db),
        "stats":       copy.deepcopy(_ds),
        "factors_l2":  copy.deepcopy(_df),  "biases_l2": copy.deepcopy(_db),
        "stats_l2":    copy.deepcopy(_ds),
        "factors_l3":  copy.deepcopy(_df),  "biases_l3": copy.deepcopy(_db),
        "stats_l3":    copy.deepcopy(_ds),
        "factors_l4":  copy.deepcopy(_df),  "biases_l4": copy.deepcopy(_db),
        "stats_l4":    copy.deepcopy(_ds),
        "dow_factors": copy.deepcopy(_ddowf),
        "dow_biases":  copy.deepcopy(_ddowb),
        "history":     [],
    }


def save_learning(sorteo: str, data: dict):
    with open(learning_path(sorteo), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_recent_sorteos(sorteo: str, n: int = 8) -> list:
    """Carga las últimas n fechas del CSV para la tabla de ocurrencias en el HTML."""
    import csv as _csv
    csv_path = os.path.join(BASE_DIR, "tombolas.csv")
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.reader(f):
            if len(row) < 22:
                continue
            if row[1].strip().upper() != sorteo.upper():
                continue
            d = row[0].strip().strip('"')[:10]  # YYYY-MM-DD
            try:
                nums = [f"{int(row[i].strip()):02d}" for i in range(2, 22)]
            except ValueError:
                continue
            rows.append({"date": d, "numbers": nums})
    return rows[-n:]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _top_candidates(input_numbers: list, transitions: dict,
                    window_key: str, top_n: int) -> set:
    raw = defaultdict(float)
    for num_str in input_numbers:
        nk    = f"{int(num_str):02d}"
        entry = transitions["transitions"].get(nk, {})
        for candidate, pct, _ in entry.get(window_key, []):
            if candidate != nk:
                raw[candidate] += pct
    if not raw:
        return set()
    return set(sorted(raw, key=raw.get, reverse=True)[:top_n])


def _pairwise_lift(a: str, b: str, cooccur: dict) -> float:
    lift_ab = next((l for n, l, _ in cooccur.get(a, []) if n == b), 1.0)
    lift_ba = next((l for n, l, _ in cooccur.get(b, []) if n == a), 1.0)
    return (lift_ab + lift_ba) / 2.0


def _group_joint_score(nums: tuple, score_map: dict, cooccur: dict) -> float:
    ind_sum  = sum(score_map.get(n, 0.0) for n in nums)
    pairs    = list(combinations(nums, 2))
    avg_lift = sum(_pairwise_lift(a, b, cooccur) for a, b in pairs) / max(1, len(pairs))
    boost    = max(0.0, avg_lift - 1.0) * COOCCUR_INFLUENCE
    return ind_sum * (1.0 + boost)


def _extract_used_from_groups(groups: dict) -> set:
    """Devuelve el conjunto de números usados en E1-E4 (excluye 'resto')."""
    used = set()
    for i in range(1, NUM_GROUPS + 1):
        for n, _ in groups.get(f"elite_{i}", []):
            used.add(n)
    return used


def build_groups_cooccur(ranked: list, cooccur: dict) -> list:
    candidates = ranked[:COOCCUR_TOP_N]
    score_map  = {n: s for n, s in candidates}
    nums       = [n for n, _ in candidates]

    scored = []
    for grp in combinations(nums, COOCCUR_GROUP_SIZE):
        js       = _group_joint_score(grp, score_map, cooccur)
        pairs    = list(combinations(grp, 2))
        avg_lift = sum(_pairwise_lift(a, b, cooccur) for a, b in pairs) / max(1, len(pairs))
        scored.append((js, avg_lift, grp))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    used     = set()
    for js, avg_lift, grp in scored:
        if not any(n in used for n in grp):
            selected.append((js, avg_lift, grp))
            used.update(grp)
        if len(selected) == NUM_COOCCUR_GROUPS:
            break

    return selected


# ──────────────────────────────────────────────
# Scoring (idéntico a v1.1)
# ──────────────────────────────────────────────

def compute_scores_v12(input_numbers: list[str],
                       transitions: dict,
                       factors: dict,
                       biases: dict,
                       dow: int | None = None) -> dict[str, float]:
    # Paso 1: Blend multi-ventana (v1.2 incluye next_day2_all y next_day3_all)
    active_weights = {}
    for w_key, w_val in WINDOW_WEIGHTS.items():
        has_data = False
        for num_str in input_numbers[:8]:
            num_key    = f"{int(num_str):02d}"
            entry_data = transitions["transitions"].get(num_key, {}).get(w_key, [])
            if entry_data and any(item[2] > 0 for item in entry_data[:10]):
                has_data = True
                break
        if has_data:
            active_weights[w_key] = w_val

    # Si faltan ventanas, redistribuir peso a next_day_all
    if "next_day_all" in active_weights and len(active_weights) < len(WINDOW_WEIGHTS):
        missing = sum(v for k, v in WINDOW_WEIGHTS.items() if k not in active_weights)
        active_weights["next_day_all"] = active_weights.get("next_day_all", 0) + missing

    if "next_day_5s" in active_weights and "next_day_30d" in active_weights:
        top_5s  = _top_candidates(input_numbers, transitions, "next_day_5s",  ADAPTIVE_TOP_N)
        top_30d = _top_candidates(input_numbers, transitions, "next_day_30d", ADAPTIVE_TOP_N)
        overlap = len(top_5s & top_30d) / ADAPTIVE_TOP_N if (top_5s and top_30d) else 0.0
        if overlap < ADAPTIVE_THRESHOLD:
            active_weights["next_day_all"] = (active_weights.get("next_day_all", 0)
                                              + active_weights.pop("next_day_5s"))

    total_w = sum(active_weights.values())

    raw = defaultdict(float)
    for num_str in input_numbers:
        num_key = f"{int(num_str):02d}"
        entry   = transitions["transitions"].get(num_key)
        if not entry:
            continue
        for w_key, w_val in active_weights.items():
            for candidate, pct, _ in entry.get(w_key, []):
                if candidate == num_key:
                    continue
                raw[candidate] += pct * (w_val / total_w)

    n = len(input_numbers)
    base = {f"{i:02d}": raw.get(f"{i:02d}", 0.0) / n for i in range(100)}

    # Paso 2: Ajuste DOW
    if dow is not None:
        for k in base:
            dow_vec = transitions["transitions"].get(k, {}).get("dow_bias", [1.0] * 7)
            bias_d  = dow_vec[dow] if dow < len(dow_vec) else 1.0
            base[k] *= (1.0 + (bias_d - 1.0) * DOW_INFLUENCE)

    # Paso 3: Cold/Carryover/Skip-day
    input_set = set(input_numbers)
    for k in base:
        if k in input_set:
            effective_dsl = 0
        else:
            dsl_json = transitions["transitions"].get(k, {}).get("days_since_last")
            if dsl_json is None:
                continue
            effective_dsl = 1 if dsl_json == 0 else dsl_json
        if effective_dsl >= COLD_THRESHOLD:
            factor = min(effective_dsl / (COLD_THRESHOLD * 3), 1.0)
            base[k] *= (1.0 + COLD_BOOST * factor)
        elif effective_dsl == 0:
            lift_factor = factors.get(k, 1.0)
            base[k] *= (1.0 + CARRYOVER_BOOST * lift_factor)
        elif effective_dsl == 1:
            base[k] *= (1.0 + SKIPDAY_BOOST)

    # Paso 3b: Decil bias estático
    for k in base:
        db = transitions["transitions"].get(k, {}).get("decil_bias", 1.0)
        base[k] *= db

    # Paso 3c: Concentración de decil en input
    input_decil_cnt: dict[int, int] = {}
    for n in input_numbers:
        d = int(n) // 10
        input_decil_cnt[d] = input_decil_cnt.get(d, 0) + 1

    for k in base:
        d   = int(k) // 10
        cnt = input_decil_cnt.get(d, 0)
        if cnt >= INPUT_DECIL_CONC_THR:
            excess   = cnt - INPUT_DECIL_CONC_THR + 1
            base[k] *= (1.0 + INPUT_DECIL_CONC_INFLUENCE * excess)

    # Paso 4: Blend rank [0,1] + score normalizado
    sorted_nums = sorted(base, key=base.get, reverse=True)
    rank_score  = {num: 1.0 - (i / 99.0) for i, num in enumerate(sorted_nums)}

    min_s = min(base.values())
    max_s = max(base.values())
    rng   = max_s - min_s if max_s > min_s else 1.0
    norm_raw = {k: (v - min_s) / rng for k, v in base.items()}

    RANK_BLEND = 0.5
    blended = {k: RANK_BLEND * rank_score[k] + (1.0 - RANK_BLEND) * norm_raw[k]
               for k in base}

    # Paso 5: Lift × bias
    final = {}
    for k, rs in blended.items():
        f = factors.get(k, 1.0)
        b = biases.get(k, 0.0)
        final[k] = round(max(0.0, rs * f + b), 5)

    return final


def rank_candidates(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def build_groups(ranked: list[tuple[str, float]],
                 input_numbers: list[str] | None = None,
                 stats: dict | None = None) -> dict:
    groups: dict = {}
    used: set = set()

    for i in range(NUM_GROUPS):
        key   = f"elite_{i + 1}"
        group = []
        decil_count: dict[int, int] = {}
        for num, score in ranked:
            if num in used:
                continue
            d = int(num) // 10
            if decil_count.get(d, 0) >= MAX_PER_DECIL:
                continue
            group.append((num, score))
            decil_count[d] = decil_count.get(d, 0) + 1
            used.add(num)
            if len(group) == GROUP_SIZE:
                break
        groups[key] = group

    # Carryover mínimo garantizado en E1+E2
    if input_numbers and MIN_CARRYOVER_SLOTS > 0:
        input_set  = set(input_numbers)
        betting    = groups["elite_1"] + groups["elite_2"]
        cy_in      = sum(1 for n, _ in betting if n in input_set)
        slots_need = MIN_CARRYOVER_SLOTS - cy_in

        if slots_need > 0:
            cy_candidates = [(n, s) for n, s in ranked
                             if n in input_set and n not in used]
            e2 = groups["elite_2"]
            e2_non_cy = sorted(
                [(idx, n, s) for idx, (n, s) in enumerate(e2) if n not in input_set],
                key=lambda x: x[2]
            )
            for (cy_num, cy_score), (idx, old_num, _) in zip(
                    cy_candidates[:slots_need], e2_non_cy[:slots_need]):
                e2[idx] = (cy_num, cy_score)
                used.add(cy_num)
                used.discard(old_num)
            groups["elite_2"] = e2

    # High-lift mínimo garantizado en E1+E2
    if stats is not None and MIN_HIGHLIFT_SLOTS > 0:
        betting     = groups["elite_1"] + groups["elite_2"]
        factors_ref = stats.get("__factors__", {})

        def _qualifies_hl(n: str) -> bool:
            s2   = stats.get(n, {})
            lift = factors_ref.get(n, 1.0)
            return lift >= HIGHLIFT_THR and s2.get("predicted", 0) >= HIGHLIFT_MIN_PRED

        hl_in      = sum(1 for n, _ in betting if _qualifies_hl(n))
        slots_need = MIN_HIGHLIFT_SLOTS - hl_in

        if slots_need > 0:
            hl_candidates = [(n, s) for n, s in ranked
                             if _qualifies_hl(n) and n not in used]
            e2 = groups["elite_2"]
            e2_non_hl = sorted(
                [(idx, n, s) for idx, (n, s) in enumerate(e2) if not _qualifies_hl(n)],
                key=lambda x: x[2]
            )
            for (hl_num, hl_score), (idx, old_num, _) in zip(
                    hl_candidates[:slots_need], e2_non_hl[:slots_need]):
                e2[idx] = (hl_num, hl_score)
                used.add(hl_num)
                used.discard(old_num)
            groups["elite_2"] = e2

    groups["resto"] = [(n, s) for n, s in ranked if n not in used]
    return groups


# ──────────────────────────────────────────────
# Aprendizaje v1.2 — factores por capa
# ──────────────────────────────────────────────

def _compute_factors_from_weighted(wp: dict, wh: dict, appeared: dict) -> tuple[dict, dict, dict]:
    """Calcula (factors, biases, stats) desde acumuladores ponderados."""
    factors: dict = {}
    biases:  dict = {}
    stats:   dict = {}
    for n in range(100):
        k  = f"{n:02d}"
        wp_ = wp.get(k, 0.0)
        wh_ = wh.get(k, 0.0)
        precision = (wh_ + LAPLACE_K * BASE_RATE) / (wp_ + LAPLACE_K)
        lift = round(max(MIN_FACTOR, min(MAX_FACTOR, precision / BASE_RATE)), 4)
        bias = round(max(MIN_BIAS,   min(MAX_BIAS,  (precision - BASE_RATE) * BIAS_SCALE)), 4)
        factors[k] = lift
        biases[k]  = bias
        stats[k]   = {
            "predicted":   int(wp_ + 0.5),
            "appeared":    appeared.get(k, 0),
            "hits":        int(wh_ + 0.5),
            "w_predicted": round(wp_, 2),
            "w_hits":      round(wh_, 2),
        }
    return factors, biases, stats


def recompute_learning_v12(history: list[dict]) -> tuple:
    """
    Recalcula factores para las 4 capas de v1.2 con decay temporal.

    Retorna:
      (factors, biases, stats,
       factors_l2, biases_l2, stats_l2,
       factors_l3, biases_l3, stats_l3,
       factors_l4, biases_l4, stats_l4,
       dow_factors, dow_biases)
    """
    completed = [h for h in history
                 if h.get("actual") and h.get("predicted_groups")]
    total = len(completed)

    # Acumuladores por capa
    wp1 = defaultdict(float); wh1 = defaultdict(float)
    wp2 = defaultdict(float); wh2 = defaultdict(float)
    wp3 = defaultdict(float); wh3 = defaultdict(float)
    wp4 = defaultdict(float); wh4 = defaultdict(float)
    appeared: dict = defaultdict(int)

    for i, entry in enumerate(completed):
        w          = math.exp(-DECAY * (total - 1 - i) / max(1, total - 1))
        actual_set = set(entry["actual"])
        for num in actual_set:
            appeared[num] += 1

        for nums in entry.get("predicted_groups", {}).values():
            for num in nums:
                wp1[num] += w
                if num in actual_set:
                    wh1[num] += w

        for nums in entry.get("predicted_groups_l2", {}).values():
            for num in nums:
                wp2[num] += w
                if num in actual_set:
                    wh2[num] += w

        for nums in entry.get("predicted_groups_l3", {}).values():
            for num in nums:
                wp3[num] += w
                if num in actual_set:
                    wh3[num] += w

        for nums in entry.get("predicted_groups_l4", {}).values():
            for num in nums:
                wp4[num] += w
                if num in actual_set:
                    wh4[num] += w

    factors,    biases,    stats    = _compute_factors_from_weighted(wp1, wh1, appeared)
    factors_l2, biases_l2, stats_l2 = _compute_factors_from_weighted(wp2, wh2, appeared)
    factors_l3, biases_l3, stats_l3 = _compute_factors_from_weighted(wp3, wh3, appeared)
    factors_l4, biases_l4, stats_l4 = _compute_factors_from_weighted(wp4, wh4, appeared)

    # DOW factors — calculados solo sobre Capa 1 (más datos)
    dow_factors = {str(d): dict(factors) for d in range(7)}
    dow_biases  = {str(d): dict(biases)  for d in range(7)}

    by_dow: dict = defaultdict(list)
    for entry in completed:
        d = entry.get("dow")
        if d is not None:
            by_dow[str(d)].append(entry)

    for d_str, d_entries in by_dow.items():
        if len(d_entries) < MIN_DOW_ROUNDS:
            continue
        total_d = len(d_entries)
        wp_d = defaultdict(float)
        wh_d = defaultdict(float)
        for i, entry in enumerate(d_entries):
            w = math.exp(-DECAY * (total_d - 1 - i) / max(1, total_d - 1))
            actual_set = set(entry["actual"])
            for nums in entry.get("predicted_groups", {}).values():
                for num in nums:
                    wp_d[num] += w
                    if num in actual_set:
                        wh_d[num] += w
        for n in range(100):
            k  = f"{n:02d}"
            wp = wp_d.get(k, 0.0)
            wh = wh_d.get(k, 0.0)
            precision = (wh + LAPLACE_K * BASE_RATE) / (wp + LAPLACE_K)
            lift = round(max(MIN_FACTOR, min(MAX_FACTOR, precision / BASE_RATE)), 4)
            bias = round(max(MIN_BIAS,   min(MAX_BIAS,   (precision - BASE_RATE) * BIAS_SCALE)), 4)
            dow_factors[d_str][k] = lift
            dow_biases[d_str][k]  = bias

    return (factors,    biases,    stats,
            factors_l2, biases_l2, stats_l2,
            factors_l3, biases_l3, stats_l3,
            factors_l4, biases_l4, stats_l4,
            dow_factors, dow_biases)


# ──────────────────────────────────────────────
# Selección compacta (2 por capa)
# ──────────────────────────────────────────────

_COMPACT_CONF = {
    1: ("★★★", "ALTA CONFIANZA ", "Capa 1 — Principal   "),
    2: ("★★ ", "CONFIANZA MEDIA", "Capa 2 — Secundaria  "),
    3: ("★  ", "BAJA CONFIANZA ", "Capa 3 — Terciaria   "),
    4: ("·  ", "ZONA RESIDUAL  ", "Capa 4 — Cuaternaria "),
}


def print_compact_selection(groups1: dict, groups2: dict, groups3: dict, groups4: dict,
                             factors_l1: dict, factors_l2: dict,
                             factors_l3: dict, factors_l4: dict) -> None:
    """
    Muestra 2 números por capa (8 en total) ordenados por confianza.

    Cada número viene con score (señal de transición + lift aprendido)
    y el lift individual del modelo para esa capa.

    Criterio de selección: top 2 del Grupo A de cada capa
    (los números con mayor score compuesto dentro de su capa).
    """
    print()
    print("  " + "═" * 62)
    print("  SELECCION COMPACTA  —  2 numeros por capa  (8 en total)")
    print("  " + "═" * 62)

    layer_data = [
        (1, groups1, factors_l1),
        (2, groups2, factors_l2),
        (3, groups3, factors_l3),
        (4, groups4, factors_l4),
    ]

    all8: list[str] = []
    for layer_num, groups, facts in layer_data:
        stars, conf_label, capa_label = _COMPACT_CONF[layer_num]
        elite = groups.get("elite_1", [])
        top2  = elite[:2]
        if len(top2) < 2:
            continue

        n1, s1 = top2[0]; l1 = facts.get(n1, 1.0)
        n2, s2 = top2[1]; l2 = facts.get(n2, 1.0)
        all8.extend([n1, n2])

        s1_mk = "★" if s1 >= SCORE_HIGH else ("⚠" if s1 < SCORE_THRESHOLD else " ")
        s2_mk = "★" if s2 >= SCORE_HIGH else ("⚠" if s2 < SCORE_THRESHOLD else " ")

        print(f"  {stars}  {conf_label}  ({capa_label})")
        print(f"       {n1}{s1_mk}  score={s1:.3f}  lift={l1:.3f}"
              f"    {n2}{s2_mk}  score={s2:.3f}  lift={l2:.3f}")

    print()
    print(f"  COMPACTO:  {' · '.join(all8)}")
    print("  " + "═" * 62)
    print()


# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────

GROUP_LABELS = {
    "elite_1": ("🥇", "GRUPO A", "máxima confianza — APOSTAR"),
    "elite_2": ("🥈", "GRUPO B", "alta confianza — APOSTAR"),
    "elite_3": ("🎯", "GRUPO C", "confianza media — APOSTAR"),
    "elite_4": ("🔵", "GRUPO D", "confianza base — APOSTAR"),
}
DOW_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
LAYER_NAMES = {1: "PRINCIPAL", 2: "SECUNDARIA", 3: "TERCIARIA", 4: "CUATERNARIA"}


def fmt_group(items: list[tuple[str, float]], cols: int = 6, threshold: float = 0.0) -> str:
    lines, row = [], []
    for i, (num, score) in enumerate(items):
        if threshold > 0 and score < threshold:
            marker = "⚠"
        elif score >= SCORE_HIGH:
            marker = "★"
        else:
            marker = ""
        row.append(f"{num}{marker}({score:.3f})")
        if (i + 1) % cols == 0:
            lines.append("  " + "  ".join(row)); row = []
    if row:
        lines.append("  " + "  ".join(row))
    return "\n".join(lines)


def fmt_plain(items: list[tuple[str, float]], cols: int = 6) -> str:
    nums  = [n for n, _ in items]
    lines = []
    for i in range(0, len(nums), cols):
        lines.append("  " + "  ".join(nums[i:i+cols]))
    return "\n".join(lines)


def print_betting_groups(groups: dict) -> None:
    e1 = groups.get("elite_1", [])
    e2 = groups.get("elite_2", [])
    e3 = groups.get("elite_3", [])
    e4 = groups.get("elite_4", [])

    def tag(num: str, score: float) -> str:
        if score < SCORE_THRESHOLD:
            return f"{num}⚠"
        if score >= SCORE_HIGH:
            return f"{num}★"
        return num

    print("  ── Grupos Calculados ─────────────────────────────────────────")
    print()
    print("  Verticales [A | B | C | D]")
    for i in range(GROUP_SIZE):
        row_tags = []
        for elite in [e1, e2, e3, e4]:
            if i < len(elite):
                n, s = elite[i]
                row_tags.append(f"{tag(n, s)}({s:.3f})")
            else:
                row_tags.append("--")
        print(f"    V{i+1}: [{', '.join(row_tags)}]")

    print()
    print("  Combinados A+B — APOSTAR:")
    for i in range(0, GROUP_SIZE, 2):
        row = []
        for elite in [e1, e2]:
            for j in [i, i + 1]:
                if j < len(elite):
                    row.append(elite[j])
        ci    = i // 2 + 1
        parts = [
            f"{n}({s:.3f})" + ("⚠" if s < SCORE_THRESHOLD else ("★" if s >= SCORE_HIGH else ""))
            for n, s in row
        ]
        n_weak = sum(1 for _, s in row if s < SCORE_THRESHOLD)
        flag   = f"  [{n_weak} débil{'es' if n_weak > 1 else ''}]" if n_weak else ""
        print(f"    C{ci}: [{', '.join(parts)}]{flag}")

    print()
    print("  Combinados C+D — APOSTAR:")
    for i in range(0, GROUP_SIZE, 2):
        row = []
        for elite in [e3, e4]:
            for j in [i, i + 1]:
                if j < len(elite):
                    row.append(elite[j])
        ci    = i // 2 + 4
        parts = [
            f"{n}({s:.3f})" + ("⚠" if s < SCORE_THRESHOLD else ("★" if s >= SCORE_HIGH else ""))
            for n, s in row
        ]
        n_weak = sum(1 for _, s in row if s < SCORE_THRESHOLD)
        flag   = f"  [{n_weak} débil{'es' if n_weak > 1 else ''}]" if n_weak else ""
        print(f"    C{ci}: [{', '.join(parts)}]{flag}")

    print()
    print(f"  Leyenda: ★ score ≥ {SCORE_HIGH}  ⚠ score < {SCORE_THRESHOLD} (débil)")


def print_cooccur_groups(cooccur_groups: list, score_map: dict, cooccur: dict) -> None:
    print()
    print("  " + "=" * 58)
    print("  GRUPOS CO-OCURRENCIA — objetivo: predecir los 4 del grupo")
    print("  " + "=" * 58)
    labels = ["CG1 (principal)", "CG2", "CG3"]
    for i, (js, avg_lift, grp) in enumerate(cooccur_groups):
        nums_str = "  ".join(grp)
        tag_str  = "  ★ APOSTAR" if i == 0 else ""
        print(f"  {labels[i]}: {nums_str}   lift_avg={avg_lift:.3f}{tag_str}")
        pairs = sorted(combinations(grp, 2),
                       key=lambda ab: _pairwise_lift(ab[0], ab[1], cooccur),
                       reverse=True)
        pair_strs = [f"[{a}-{b}]({_pairwise_lift(a,b,cooccur):.2f}x)" for a, b in pairs[:3]]
        print(f"           pares: {' '.join(pair_strs)}")
    print()
    print("  Para registrar 4 aciertos exactos en un grupo:")
    print("  Aposta CG1 completo como unidad (los 4 números).")
    print("  " + "=" * 58)


def print_prediction_header(sorteo: str, input_numbers: list, dow: int | None,
                             learning: dict) -> None:
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"
    rondas = len(learning["history"])
    dow_txt = (f"  Día del sorteo: {DOW_NAMES[dow]} (DOW={dow})" if dow is not None
               else "  Día del sorteo: no especificado (pasar --date YYYY-MM-DD)")
    print()
    print("=" * 62)
    print(f"  PRONÓSTICO v1.2 — TOMBOLA {s_name.upper()} ({sorteo})")
    print(f"  Análisis: transitions_v12 | Entrenamiento: learning_v12")
    print("=" * 62)
    print(f"  Entrada ({len(input_numbers)} números): {' '.join(sorted(input_numbers))}")
    print(dow_txt)
    print(f"  Rondas aprendidas: {rondas}")


def print_layer(layer_num: int, groups: dict, show_scores: bool,
                cooccur_groups=None, score_map=None, cooccur=None) -> None:
    """Imprime una capa completa: A/B/C/D (todos APOSTAR), verticales, combinados, co-oc."""
    name = LAYER_NAMES.get(layer_num, f"CAPA {layer_num}")
    print()
    print("  " + "═" * 58)
    print(f"  CAPA {layer_num} — {name}  (24 números)")
    print("  " + "═" * 58)
    print()

    for i in range(NUM_GROUPS):
        key   = f"elite_{i + 1}"
        items = groups.get(key, [])
        emoji, label, desc = GROUP_LABELS[key]
        print(f"  {emoji} {label}  ({GROUP_SIZE} números — {desc})")
        print(fmt_group(items, cols=6, threshold=SCORE_THRESHOLD) if show_scores
              else fmt_plain(items, cols=6))
        print()

    print_betting_groups(groups)

    if cooccur_groups and score_map is not None and cooccur is not None:
        print_cooccur_groups(cooccur_groups, score_map, cooccur)


# ──────────────────────────────────────────────
# Generación HTML
# ──────────────────────────────────────────────

def _build_layer_data(groups: dict, cooccur_groups: list | None) -> dict:
    """Convierte la salida de build_groups en estructura JSON para el HTML."""
    e1 = groups.get("elite_1", [])
    e2 = groups.get("elite_2", [])
    e3 = groups.get("elite_3", [])
    e4 = groups.get("elite_4", [])

    def item(n, s):
        return {"num": n, "score": round(s, 3)}

    verticals = []
    for i in range(GROUP_SIZE):
        row = {}
        for key, lbl in [("elite_1","A"),("elite_2","B"),("elite_3","C"),("elite_4","D")]:
            g = groups.get(key, [])
            if i < len(g):
                row[lbl] = item(*g[i])
        verticals.append(row)

    def combos(elites, base_idx):
        out = []
        for i in range(0, GROUP_SIZE, 2):
            nums = [item(*elites[k][j]) for k in range(len(elites))
                    for j in [i, i+1] if j < len(elites[k])]
            out.append({"idx": base_idx + i//2, "nums": nums})
        return out

    cooc = []
    if cooccur_groups:
        for i, (_, avg_lift, grp) in enumerate(cooccur_groups):
            cooc.append({"label": f"CG{i+1}", "nums": list(grp),
                         "lift": round(avg_lift, 3), "is_main": i == 0})

    return {
        "groups": {"A": [item(*x) for x in e1], "B": [item(*x) for x in e2],
                   "C": [item(*x) for x in e3], "D": [item(*x) for x in e4]},
        "verticals": verticals,
        "combinados_ab": combos([e1, e2], 1),
        "combinados_cd": combos([e3, e4], 4),
        "cooccur": cooc,
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tombola v1.2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#f8fafc;padding-bottom:72px}
.hdr{background:#1e293b;border-bottom:2px solid #334155;padding:14px 20px;position:sticky;top:0;z-index:100}
.hdr h1{font-size:1.25rem;font-weight:700;color:#fbbf24}
.hdr .meta{font-size:.75rem;color:#94a3b8;margin-top:3px}
.hdr .inp{font-size:.7rem;color:#64748b;font-family:monospace;margin-top:2px;word-break:break-all}
.tabs{display:flex;gap:8px;padding:12px 20px;background:#0f172a;border-bottom:1px solid #1e293b;flex-wrap:wrap}
.tb{padding:7px 16px;border-radius:8px;border:2px solid #334155;cursor:pointer;font-weight:600;font-size:.8rem;background:#1e293b;color:#94a3b8;transition:.15s}
.tb.on[data-l="1"]{border-color:#fbbf24;color:#fbbf24;background:#422006}
.tb.on[data-l="2"]{border-color:#94a3b8;color:#e2e8f0;background:#1e293b}
.tb.on[data-l="3"]{border-color:#cd7c2f;color:#cd7c2f;background:#321d05}
.tb.on[data-l="4"]{border-color:#22d3ee;color:#22d3ee;background:#082f3e}
.lc{display:none;padding:16px 20px}
.lc.on{display:block}
.lh{font-size:1.1rem;font-weight:700;margin-bottom:14px;color:#fbbf24}
.gg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.gc{background:#1e293b;border-radius:10px;overflow:hidden}
.gh{padding:9px 10px;cursor:pointer;display:flex;align-items:center;gap:7px}
.gh:hover{filter:brightness(1.12)}
.gh .em{font-size:1.1rem}
.gh .gl{font-weight:700;font-size:.8rem}
.gh .gd{font-size:.65rem;color:#94a3b8}
.gn{padding:10px;display:flex;flex-direction:column;gap:6px}
.nc{display:flex;align-items:center;gap:6px;padding:5px 8px;border-radius:7px;cursor:pointer;background:#0f172a;border:2px solid transparent;transition:.12s}
.nc:hover{border-color:#475569}
.nc.sel{border-color:#fbbf24!important}
.nc .nb{font-size:1rem;font-weight:700;font-family:monospace;width:26px;text-align:center}
.nc .sb{font-size:.65rem;padding:2px 5px;border-radius:3px;font-family:monospace}
.sh{background:#78350f;color:#fef3c7}
.so{background:#1e3a8a;color:#bfdbfe}
.sw{background:#374151;color:#9ca3af}
.sec{background:#1e293b;border-radius:10px;padding:14px;margin-bottom:12px}
.st{font-size:.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px}
table.vt{width:100%;border-collapse:collapse}
table.vt th{padding:7px 8px;font-size:.7rem;color:#94a3b8;border-bottom:1px solid #334155}
table.vt td{padding:5px 6px;border-bottom:1px solid #0f172a;text-align:center}
.vn{display:inline-flex;align-items:center;gap:3px;padding:3px 7px;border-radius:6px;cursor:pointer;font-family:monospace;font-weight:700;font-size:.85rem;border:1px solid transparent}
.vn.sel{border-color:#fbbf24}
.cg{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.cc{background:#0f172a;border-radius:8px;padding:11px;cursor:pointer}
.cc:hover{background:#162032}
.cl{font-size:.7rem;font-weight:700;color:#64748b;display:flex;justify-content:space-between;margin-bottom:7px}
.cn{display:flex;gap:5px;flex-wrap:wrap}
.cb{width:34px;height:34px;display:flex;align-items:center;justify-content:center;border-radius:7px;font-family:monospace;font-weight:700;font-size:.8rem;cursor:pointer;border:2px solid transparent}
.cb.sel{border-color:#fbbf24}
.qg{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.qc{background:#0f172a;border-radius:8px;padding:11px;cursor:pointer;border:1px solid #334155}
.qc.main{border-color:#fbbf24}
.qc:hover{background:#162032}
.ql{font-size:.75rem;font-weight:700;margin-bottom:3px}
.qc.main .ql{color:#fbbf24}
.qlift{font-size:.65rem;color:#64748b;margin-bottom:7px}
.qn{display:flex;gap:5px}
.qb{width:34px;height:34px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:#1e3a8a;font-family:monospace;font-weight:700;font-size:.85rem;cursor:pointer;border:2px solid transparent}
.qb.sel{border-color:#fbbf24}
.selpanel{position:fixed;bottom:0;left:0;right:0;background:#1e293b;border-top:2px solid #334155;padding:10px 20px;z-index:200}
.selinner{display:flex;align-items:center;gap:12px;max-width:1400px;margin:0 auto}
.selnums{display:flex;gap:5px;flex-wrap:wrap;flex:1}
.sc2{padding:3px 9px;background:#334155;border-radius:16px;font-family:monospace;font-weight:600;font-size:.8rem;cursor:pointer}
.sc2:hover{background:#475569}
.selcnt{font-size:.8rem;color:#94a3b8;white-space:nowrap}
.btn{padding:7px 14px;border-radius:7px;border:none;cursor:pointer;font-weight:600;font-size:.8rem}
.bcl{background:#374151;color:#f8fafc}
.bcp{background:#1d4ed8;color:#fff}
.btn:hover{filter:brightness(1.1)}
.resto{padding:16px 20px;color:#475569;font-size:.8rem}
.ot{width:100%;border-collapse:collapse;font-size:.75rem}
.ot th{padding:4px 6px;text-align:center;color:#64748b;font-weight:600;border-bottom:1px solid #334155;font-family:monospace;white-space:nowrap}
.ot th.occ-hdr-num{text-align:left;padding-left:10px;color:#94a3b8}
.ot th.occ-hdr-sum{color:#fbbf24}
.ot td{padding:4px 5px;text-align:center;border-bottom:1px solid #0f172a;font-family:monospace}
.ot td.occ-num{text-align:left;padding-left:10px;font-weight:700;cursor:pointer}
.ot td.occ-num:hover{color:#fbbf24}
.occ-hi{background:#1e3a5f33;color:#60a5fa}
.occ-lo{color:#334155}
.occ-sum{color:#fbbf24;font-weight:700}
.occ-sum0{color:#334155}
.occ-sep td{border-top:2px solid #1e293b;background:#0a111e;font-size:.65rem;color:#475569;font-family:sans-serif;font-weight:600;text-align:left;padding-left:10px}
@media(max-width:800px){.gg{grid-template-columns:repeat(2,1fr)}.cg,.qg{grid-template-columns:1fr}}
@media print{.selpanel,.tabs{display:none}.lc{display:block!important}}
</style>
</head>
<body>
<div class="hdr">
  <h1 id="h-title"></h1>
  <div class="meta" id="h-meta"></div>
  <div class="inp" id="h-inp"></div>
</div>
<div class="tabs" id="tabs"></div>
<div id="layers"></div>
<div class="resto" id="resto"></div>
<div class="selpanel">
  <div class="selinner">
    <div class="selnums" id="seld"><span style="color:#475569">Click en cualquier número para seleccionarlo</span></div>
    <span class="selcnt" id="selcnt">0 / 20</span>
    <button class="btn bcl" onclick="clearSel()">Limpiar</button>
    <button class="btn bcp" onclick="copySel()">Copiar</button>
  </div>
</div>
<script>
const D = DATA_JSON_PLACEHOLDER;
const sel = new Set();

const GBG  = {A:'#7c2d12',B:'#1e3a5f',C:'#064e3b',D:'#3b1f6e'};
const GBD  = {A:'#f97316',B:'#60a5fa',C:'#34d399',D:'#a78bfa'};
const GEM  = {A:'🥇',B:'🥈',C:'🎯',D:'🔵'};
const GDC  = {A:'máxima confianza',B:'alta confianza',C:'confianza media',D:'confianza base'};

function scl(s){ return s>=1.5?'sh':s>=1.05?'so':'sw'; }
function smk(s){ return s>=1.5?'★':s<1.05?'⚠':''; }

function tog(n){
  sel.has(n)?sel.delete(n):sel.add(n);
  document.querySelectorAll(`[data-n="${n}"]`).forEach(e=>e.classList.toggle('sel',sel.has(n)));
  renderSel();
}
function togGroup(nums){
  const all=nums.every(n=>sel.has(n));
  nums.forEach(n=>{all?sel.delete(n):sel.add(n);
    document.querySelectorAll(`[data-n="${n}"]`).forEach(e=>e.classList.toggle('sel',sel.has(n)));});
  renderSel();
}
function renderSel(){
  const d=document.getElementById('seld'),c=document.getElementById('selcnt');
  c.textContent=sel.size+' / 20';
  if(!sel.size){d.innerHTML='<span style="color:#475569">Click en cualquier número para seleccionarlo</span>';return;}
  d.innerHTML=[...sel].sort().map(n=>`<span class="sc2" onclick="tog('${n}')">${n}</span>`).join('');
}
function clearSel(){
  sel.forEach(n=>document.querySelectorAll(`[data-n="${n}"]`).forEach(e=>e.classList.remove('sel')));
  sel.clear();renderSel();
}
function copySel(){
  navigator.clipboard.writeText([...sel].sort().join(' ')).then(()=>{
    const b=document.querySelector('.bcp');b.textContent='✓ Copiado';
    setTimeout(()=>b.textContent='Copiar',1600);
  });
}
function showLayer(n){
  document.querySelectorAll('.lc').forEach(e=>e.classList.remove('on'));
  document.querySelectorAll('.tb').forEach(e=>e.classList.remove('on'));
  document.getElementById('l'+n).classList.add('on');
  document.querySelector(`.tb[data-l="${n}"]`).classList.add('on');
}

function numChip(item,g){
  const sc=scl(item.score),mk=smk(item.score);
  return `<div class="nc" data-n="${item.num}" onclick="tog('${item.num}')">
    <span class="nb">${item.num}</span>
    <span class="sb ${sc}">${item.score.toFixed(3)} ${mk}</span>
  </div>`;
}
function comboNum(item){
  const sc=scl(item.score);
  const bg=sc==='sh'?'#78350f':sc==='so'?'#1e3a8a':'#374151';
  return `<div class="cb" data-n="${item.num}" style="background:${bg}" onclick="tog('${item.num}')">${item.num}<sup style="font-size:.55rem">${smk(item.score)}</sup></div>`;
}

function renderOccTable(lay){
  if(!D.recent_sorteos||!D.recent_sorteos.length)return'';
  const rs=D.recent_sorteos;
  const sets=rs.map(s=>new Set(s.numbers));
  const fmt=d=>{const p=d.split('-');return p[2]+'/'+p[1];};
  const combos=[...lay.combinados_ab,...lay.combinados_cd];
  const nc=rs.length+2;
  let h=`<div class="sec"><div class="st">Historial de ocurrencias — últimas ${rs.length} fechas nocturnas</div><div style="overflow-x:auto"><table class="ot">`;
  h+=`<tr><th class="occ-hdr-num">Nº</th>${rs.map(s=>`<th>${fmt(s.date)}</th>`).join('')}<th class="occ-hdr-sum">Σ</th></tr>`;
  combos.forEach(combo=>{
    h+=`<tr class="occ-sep"><td colspan="${nc}">C${combo.idx}</td></tr>`;
    combo.nums.forEach(item=>{
      const n=item.num;
      const occs=sets.map(s=>s.has(n)?1:0);
      const tot=occs.reduce((a,b)=>a+b,0);
      h+=`<tr><td class="occ-num" data-n="${n}" onclick="tog('${n}')">${n}</td>`;
      h+=occs.map(o=>`<td class="${o?'occ-hi':'occ-lo'}">${o}</td>`).join('');
      h+=`<td class="${tot>0?'occ-sum':'occ-sum0'}">${tot}</td></tr>`;
    });
  });
  h+=`</table></div></div>`;
  return h;
}

function renderLayer(lay){
  const G=['A','B','C','D'];
  // Groups
  let gg=`<div class="gg">`;
  G.forEach(g=>{
    const nums=lay.groups[g].map(x=>x.num);
    gg+=`<div class="gc">
      <div class="gh" style="background:${GBG[g]}" onclick="togGroup(${JSON.stringify(nums)})">
        <span class="em">${GEM[g]}</span><div><div class="gl">GRUPO ${g}</div><div class="gd">${GDC[g]} — APOSTAR</div></div>
      </div>
      <div class="gn">${lay.groups[g].map(x=>numChip(x,g)).join('')}</div>
    </div>`;
  });
  gg+=`</div>`;

  // Verticals
  let vt=`<div class="sec"><div class="st">Verticales V1–V6 · click fila para seleccionar</div>
    <table class="vt"><tr><th>V</th>${G.map(g=>`<th>${GEM[g]} ${g}</th>`).join('')}</tr>`;
  lay.verticals.forEach((row,i)=>{
    const rn=G.filter(g=>row[g]).map(g=>row[g].num);
    vt+=`<tr onclick="togGroup(${JSON.stringify(rn)})" style="cursor:pointer"><td style="color:#64748b;font-weight:700">V${i+1}</td>`;
    G.forEach(g=>{
      const c=row[g];
      if(c) vt+=`<td><div class="vn" data-n="${c.num}" style="background:${GBG[g]}33;border-color:${GBG[g]}55">${c.num} <span style="font-size:.6rem;color:#94a3b8">${c.score.toFixed(3)}${smk(c.score)}</span></div></td>`;
      else vt+=`<td style="color:#334155">—</td>`;
    });
    vt+=`</tr>`;
  });
  vt+=`</table></div>`;

  function renderCombos(combos,title){
    let h=`<div class="sec"><div class="st">${title}</div><div class="cg">`;
    combos.forEach(c=>{
      const ns=c.nums.map(x=>x.num);
      const wk=c.nums.filter(x=>x.score<1.05).length;
      h+=`<div class="cc" onclick="togGroup(${JSON.stringify(ns)})">
        <div class="cl"><span>C${c.idx} — APOSTAR</span>${wk?`<span style="color:#d97706">${wk}⚠</span>`:''}</div>
        <div class="cn">${c.nums.map(comboNum).join('')}</div>
      </div>`;
    });
    h+=`</div></div>`;return h;
  }

  // Co-occurrence
  let qo=`<div class="sec"><div class="st">Co-Ocurrencia · click grupo para seleccionar</div><div class="qg">`;
  lay.cooccur.forEach(cg=>{
    const cls=cg.is_main?'qc main':'qc';
    qo+=`<div class="${cls}" onclick="togGroup(${JSON.stringify(cg.nums)})">
      <div class="ql">${cg.label}${cg.is_main?' ★':''}</div>
      <div class="qlift">lift_avg = ${cg.lift.toFixed(3)}</div>
      <div class="qn">${cg.nums.map(n=>`<div class="qb" data-n="${n}" onclick="event.stopPropagation();tog('${n}')">${n}</div>`).join('')}</div>
    </div>`;
  });
  qo+=`</div></div>`;

  return `<div id="l${lay.num}" class="lc${lay.num===1?' on':''}">
    <h2 class="lh">CAPA ${lay.num} — ${lay.name} (24 números)</h2>
    ${gg}${vt}
    ${renderCombos(lay.combinados_ab,'Combinados A+B — APOSTAR · click para seleccionar')}
    ${renderCombos(lay.combinados_cd,'Combinados C+D — APOSTAR · click para seleccionar')}
    ${qo}
    ${renderOccTable(lay)}
  </div>`;
}

// Header
document.getElementById('h-title').textContent=`🎱 Tombola ${D.sorteo_name.toUpperCase()} — ${D.dow} ${D.date}`;
document.getElementById('h-meta').textContent=`v1.2 · ${D.rondas} rondas aprendidas`;
document.getElementById('h-inp').textContent=`Entrada: ${D.input.join(' ')}`;

// Tabs
const tabL=['1','2','3','4'];const tabN=['PRINCIPAL','SECUNDARIA','TERCIARIA','CUATERNARIA'];
document.getElementById('tabs').innerHTML=
  tabL.map((l,i)=>`<button class="tb${l==='1'?' on':''}" data-l="${l}" onclick="showLayer('${l}')">CAPA ${l} — ${tabN[i]}</button>`).join('');

// Layers
document.getElementById('layers').innerHTML=D.layers.map(renderLayer).join('');

// Resto
document.getElementById('resto').innerHTML=
  `<b style="color:#94a3b8">⚪ RESTO FINAL (${D.resto.length} números)</b><br>
   <span style="font-family:monospace">${D.resto.join('  ')}</span>`;
</script>
</body>
</html>"""


def generate_html(sorteo: str, date_str: str, dow_name: str,
                  input_numbers: list, rondas: int,
                  layer1: dict, layer2: dict, layer3: dict, layer4: dict,
                  resto_final: list,
                  recent_sorteos: list | None = None) -> str:
    import json as _json
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"
    payload = {
        "sorteo": sorteo, "sorteo_name": s_name,
        "date": date_str, "dow": dow_name,
        "input": sorted(input_numbers), "rondas": rondas,
        "layers": [
            {"num": 1, "name": "PRINCIPAL",   **layer1},
            {"num": 2, "name": "SECUNDARIA",  **layer2},
            {"num": 3, "name": "TERCIARIA",   **layer3},
            {"num": 4, "name": "CUATERNARIA", **layer4},
        ],
        "resto": [n for n, _ in resto_final],
        "recent_sorteos": recent_sorteos or [],
    }
    return _HTML_TEMPLATE.replace("DATA_JSON_PLACEHOLDER",
                                  _json.dumps(payload, ensure_ascii=False))


# ──────────────────────────────────────────────
# Comandos
# ──────────────────────────────────────────────

def cmd_predict(args):
    sorteo   = args.sorteo.upper()
    trans    = load_transitions(sorteo)
    learning = load_learning(sorteo)

    if args.dow is not None:
        dow = args.dow
    elif args.date is not None:
        dow = datetime.strptime(args.date, "%Y-%m-%d").weekday()
    else:
        dow = None
        print("  AVISO: sin --date no se puede determinar el día del sorteo.")
        print("         Usá --date YYYY-MM-DD con la fecha del sorteo a predecir.")
        print()

    raw = args.numbers
    if len(raw) != 20:
        print(f"ERROR: Se necesitan 20 números (recibidos: {len(raw)})."); sys.exit(1)
    try:
        input_numbers = [f"{int(n):02d}" for n in raw]
    except ValueError:
        print("ERROR: Números inválidos."); sys.exit(1)

    # ── Factores DOW para Capa 1 ───────────────────────────────────────────────
    if dow is not None and str(dow) in learning.get("dow_factors", {}):
        eff_factors_l1 = learning["dow_factors"][str(dow)]
        eff_biases_l1  = learning["dow_biases"].get(str(dow), learning["biases"])
    else:
        eff_factors_l1 = learning["factors"]
        eff_biases_l1  = learning["biases"]

    # Factores propios de Capa 2, 3 y 4
    eff_factors_l2 = learning.get("factors_l2", learning["factors"])
    eff_biases_l2  = learning.get("biases_l2",  learning["biases"])
    eff_factors_l3 = learning.get("factors_l3", learning["factors"])
    eff_biases_l3  = learning.get("biases_l3",  learning["biases"])
    eff_factors_l4 = learning.get("factors_l4", learning["factors"])
    eff_biases_l4  = learning.get("biases_l4",  learning["biases"])

    # ── Capa 1: 24 números principales ────────────────────────────────────────
    scores1 = compute_scores_v12(input_numbers, trans, eff_factors_l1, eff_biases_l1, dow)
    ranked1 = rank_candidates(scores1)
    hl_stats_l1 = dict(learning.get("stats", {}))
    hl_stats_l1["__factors__"] = eff_factors_l1
    groups1 = build_groups(ranked1, input_numbers, hl_stats_l1)
    used1   = _extract_used_from_groups(groups1)

    # ── Capa 2: 24 de los 76 restantes con factores propios ───────────────────
    scores2  = compute_scores_v12(input_numbers, trans, eff_factors_l2, eff_biases_l2, dow)
    ranked2  = sorted([(n, scores2[n]) for n in scores2 if n not in used1],
                      key=lambda x: x[1], reverse=True)
    hl_stats_l2 = dict(learning.get("stats_l2", {}))
    hl_stats_l2["__factors__"] = eff_factors_l2
    groups2 = build_groups(ranked2, input_numbers, hl_stats_l2)
    used2   = used1 | _extract_used_from_groups(groups2)

    # ── Capa 3: 24 de los 52 restantes con factores propios ───────────────────
    scores3  = compute_scores_v12(input_numbers, trans, eff_factors_l3, eff_biases_l3, dow)
    ranked3  = sorted([(n, scores3[n]) for n in scores3 if n not in used2],
                      key=lambda x: x[1], reverse=True)
    hl_stats_l3 = dict(learning.get("stats_l3", {}))
    hl_stats_l3["__factors__"] = eff_factors_l3
    groups3 = build_groups(ranked3, input_numbers, hl_stats_l3)
    used3   = used2 | _extract_used_from_groups(groups3)

    # ── Capa 4: 24 de los 28 restantes con factores propios ───────────────────
    scores4  = compute_scores_v12(input_numbers, trans, eff_factors_l4, eff_biases_l4, dow)
    ranked4  = sorted([(n, scores4[n]) for n in scores4 if n not in used3],
                      key=lambda x: x[1], reverse=True)
    hl_stats_l4 = dict(learning.get("stats_l4", {}))
    hl_stats_l4["__factors__"] = eff_factors_l4
    groups4 = build_groups(ranked4, input_numbers, hl_stats_l4)
    used4   = used3 | _extract_used_from_groups(groups4)

    # ranked combinado para co-ocurrencia (usa scores de Capa 1 como referencia)
    ranked = ranked1

    # ── Resto final: 4 números ──────────────────────────────────────────────────
    resto_final = [(n, s) for n, s in ranked if n not in used4]

    # Co-ocurrencia — solo con los 24 números de cada capa
    cooccur   = trans.get("cooccur", {})
    score_map = {n: s for n, s in ranked1}
    ranked_l1 = sorted([(n, s) for n, s in ranked1 if n in used1],
                       key=lambda x: x[1], reverse=True)
    ranked_l2 = sorted([(n, scores2[n]) for n in scores2 if n in (used2 - used1)],
                       key=lambda x: x[1], reverse=True)
    ranked_l3 = sorted([(n, scores3[n]) for n in scores3 if n in (used3 - used2)],
                       key=lambda x: x[1], reverse=True)
    ranked_l4 = sorted([(n, scores4[n]) for n in scores4 if n in (used4 - used3)],
                       key=lambda x: x[1], reverse=True)
    cooccur_groups1 = build_groups_cooccur(ranked_l1, cooccur) if cooccur else None
    cooccur_groups2 = build_groups_cooccur(ranked_l2, cooccur) if cooccur else None
    cooccur_groups3 = build_groups_cooccur(ranked_l3, cooccur) if cooccur else None
    cooccur_groups4 = build_groups_cooccur(ranked_l4, cooccur) if cooccur else None

    show_scores = not args.plain

    # ── Imprimir encabezado + selección compacta + 3 capas + resto ────────────
    print_prediction_header(sorteo, input_numbers, dow, learning)
    print_compact_selection(
        groups1, groups2, groups3, groups4,
        eff_factors_l1, eff_factors_l2, eff_factors_l3, eff_factors_l4,
    )
    print_layer(1, groups1, show_scores, cooccur_groups1, score_map, cooccur)
    print_layer(2, groups2, show_scores, cooccur_groups2, score_map, cooccur)
    print_layer(3, groups3, show_scores, cooccur_groups3, score_map, cooccur)
    print_layer(4, groups4, show_scores, cooccur_groups4, score_map, cooccur)

    print()
    print("  " + "─" * 58)
    print(f"  ⚪ RESTO FINAL ({len(resto_final)} números — zona no predicha)")
    print(f"  {' '.join(n for n, _ in resto_final)}")

    print()
    print("  Para registrar el resultado:")
    print(f'  python tombola_predict_v12.py feedback --sorteo {sorteo} \\')
    print(f'    --date {args.date or date.today().isoformat()} --actual <20 numeros>')
    print("=" * 62)
    print()

    # ── Generar HTML si se pidió ──────────────────────────────────────────────
    if getattr(args, "html", None):
        dow_name = DOW_NAMES[dow] if dow is not None else "?"
        l1 = _build_layer_data(groups1, cooccur_groups1)
        l2 = _build_layer_data(groups2, cooccur_groups2)
        l3 = _build_layer_data(groups3, cooccur_groups3)
        l4 = _build_layer_data(groups4, cooccur_groups4)
        recent_sorteos = _load_recent_sorteos(sorteo, n=8)
        html_str = generate_html(
            sorteo, args.date or date.today().isoformat(), dow_name,
            input_numbers, len(learning["history"]), l1, l2, l3, l4, resto_final,
            recent_sorteos=recent_sorteos,
        )
        html_path = args.html
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html_str)
        print(f"  ✓ HTML generado: {html_path}")
        print()

    # ── Guardar historial con las 3 capas ─────────────────────────────────────
    def _pred_dict(groups):
        return {f"elite_{i+1}": [n for n, _ in groups.get(f"elite_{i+1}", [])]
                for i in range(NUM_GROUPS)}

    pending = {
        "date":                args.date or date.today().isoformat(),
        "sorteo":              sorteo,
        "dow":                 dow,
        "input":               input_numbers,
        "predicted_groups":    _pred_dict(groups1),
        "predicted_groups_l2": _pred_dict(groups2),
        "predicted_groups_l3": _pred_dict(groups3),
        "predicted_groups_l4": _pred_dict(groups4),
        "actual":              None,
        "hits_per_group":      None,
        "hits_cumulative":     None,
        "hits_l2":             None,
        "hits_cumulative_l2":  None,
        "hits_l3":             None,
        "hits_cumulative_l3":  None,
        "hits_l4":             None,
        "hits_cumulative_l4":  None,
    }
    history = learning["history"]
    idx = next((i for i, h in enumerate(history)
                if h["date"] == pending["date"] and h["sorteo"] == sorteo
                and h["actual"] is None), None)
    if idx is not None:
        history[idx] = pending
    else:
        history.append(pending)

    save_learning(sorteo, learning)
    print(f"  ✓ Predicción guardada (4 capas). Registra el resultado con `feedback`.")
    print()


def cmd_feedback(args):
    sorteo   = args.sorteo.upper()
    learning = load_learning(sorteo)

    try:
        actual = [f"{int(n):02d}" for n in args.actual]
    except ValueError:
        print("ERROR: Números inválidos."); sys.exit(1)
    if len(actual) != 20:
        print(f"ERROR: Se necesitan 20 números reales (recibidos: {len(actual)})."); sys.exit(1)

    target_date = args.date or date.today().isoformat()
    history     = learning["history"]
    idx = next((i for i, h in enumerate(history)
                if h["date"] == target_date and h["sorteo"] == sorteo
                and h["actual"] is None), None)

    if idx is None:
        print(f"AVISO: Sin predicción pendiente para {target_date} ({sorteo}). "
              f"Registrando solo resultado.")
        history.append({"date": target_date, "sorteo": sorteo, "input": [],
                        "predicted_groups": {}, "actual": actual,
                        "hits_per_group": None, "hits_cumulative": None})
        save_learning(sorteo, learning)
        return

    entry      = history[idx]
    actual_set = set(actual)

    def _hits(pred_key: str) -> tuple[dict, dict]:
        pred_groups = entry.get(pred_key, {})
        hpg  = {}; hcum = {}; cumset: set = set()
        for i in range(NUM_GROUPS):
            gk = f"elite_{i+1}"
            gs = set(pred_groups.get(gk, []))
            hpg[gk]  = len(actual_set & gs)
            cumset  |= gs
            hcum[gk] = len(actual_set & cumset)
        return hpg, hcum

    hpg1, hcum1 = _hits("predicted_groups")
    hpg2, hcum2 = _hits("predicted_groups_l2")
    hpg3, hcum3 = _hits("predicted_groups_l3")
    hpg4, hcum4 = _hits("predicted_groups_l4")

    entry.update({
        "actual":             actual,
        "hits_per_group":     hpg1,  "hits_cumulative":     hcum1,
        "hits_l2":            hpg2,  "hits_cumulative_l2":  hcum2,
        "hits_l3":            hpg3,  "hits_cumulative_l3":  hcum3,
        "hits_l4":            hpg4,  "hits_cumulative_l4":  hcum4,
    })
    history[idx] = entry

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
    })
    save_learning(sorteo, learning)

    s_name = "Nocturna" if sorteo == "N" else "Vespertina"
    print()
    print("=" * 60)
    print(f"  RESULTADO REGISTRADO — {s_name} {target_date}")
    print("=" * 60)
    print(f"  Números reales: {' '.join(sorted(actual))}")
    print()

    for layer_num, hpg, hcum, pred_key, label in [
        (1, hpg1, hcum1, "predicted_groups",    "Capa 1 PRINCIPAL"),
        (2, hpg2, hcum2, "predicted_groups_l2", "Capa 2 SECUNDARIA"),
        (3, hpg3, hcum3, "predicted_groups_l3", "Capa 3 TERCIARIA"),
        (4, hpg4, hcum4, "predicted_groups_l4", "Capa 4 CUATERNARIA"),
    ]:
        total_pred  = entry.get(pred_key, {})
        all_pred    = set(n for g in total_pred.values() for n in g)
        total_hit   = hcum.get(f"elite_{NUM_GROUPS}", 0)
        print(f"  {label}: {total_hit}/20 aciertos en los 24 predichos")
        for i in range(NUM_GROUPS):
            gk   = f"elite_{i+1}"
            h    = hpg.get(gk, 0)
            ac   = GROUP_SIZE * (i + 1)
            azar = round(ac / 100 * 20, 1)
            print(f"    {gk}: {h}/20  (acum={hcum.get(gk,0)}  azar={azar})")
        missed = sorted(actual_set - all_pred)
        if missed and layer_num == 1:
            print(f"    No predichos: {' '.join(missed)}")
        print()

    print(f"  ✓ Factores de 4 capas actualizados (DECAY={DECAY}).")
    print("=" * 60)
    print()


def cmd_accuracy(args):
    sorteo   = args.sorteo.upper()
    learning = load_learning(sorteo)
    s_name   = "Nocturna" if sorteo == "N" else "Vespertina"
    completed = [h for h in learning["history"]
                 if h.get("actual") and h.get("hits_per_group")]
    n = len(completed)
    if n == 0:
        print(f"Sin rondas completadas para {s_name}."); return

    print()
    print("=" * 62)
    print(f"  PRECISIÓN v1.2 — {s_name} ({sorteo})")
    print("=" * 62)
    print(f"  Rondas evaluadas: {n}  |  Decay temporal: {DECAY}")

    for layer_num, hpg_key, hcum_key, layer_name in [
        (1, "hits_per_group",  "hits_cumulative",    "Capa 1 PRINCIPAL"),
        (2, "hits_l2",         "hits_cumulative_l2", "Capa 2 SECUNDARIA"),
        (3, "hits_l3",         "hits_cumulative_l3", "Capa 3 TERCIARIA"),
        (4, "hits_l4",         "hits_cumulative_l4", "Capa 4 CUATERNARIA"),
    ]:
        # Solo rondas que tienen datos de esta capa
        layer_rows = [h for h in completed if h.get(hpg_key)]
        nl = len(layer_rows)
        if nl == 0:
            continue
        print()
        print(f"  ── {layer_name} ({nl} rondas) ──────────────────────────────")
        print(f"  {'Grupo':<10} {'Prom.solo':>10} {'Prom.acum':>10} {'Azar':>6}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
        for i in range(NUM_GROUPS):
            gk   = f"elite_{i+1}"
            solo = sum(h[hpg_key].get(gk, 0) for h in layer_rows) / nl
            acum = sum(h.get(hcum_key, {}).get(gk, 0) for h in layer_rows) / nl
            azar = round(GROUP_SIZE * (i + 1) / 100 * 20, 1)
            print(f"  {gk:<10} {solo:>9.2f}  {acum:>9.2f}  {azar:>6.1f}")

    # Últimas rondas (todas las capas)
    last = completed[-5:]
    hdr  = "  ".join(f"E{i+1}" for i in range(NUM_GROUPS))
    print()
    print(f"  Últimas {len(last)} rondas — hits por capa (acum E4):")
    print(f"  {'Fecha':<12}  C1    C2    C3    C4")
    for h in last:
        def _total(key):
            d = h.get(key, {})
            return str(d.get(f"elite_{NUM_GROUPS}", "-")) if d else "-"
        print(f"  {h['date']:<12}  {_total('hits_cumulative'):>4}  "
              f"{_total('hits_cumulative_l2'):>4}  {_total('hits_cumulative_l3'):>4}  "
              f"{_total('hits_cumulative_l4'):>4}")

    # Top lifts por capa
    for layer_name, fkey, skey in [
        ("Capa 1", "factors",    "stats"),
        ("Capa 2", "factors_l2", "stats_l2"),
        ("Capa 3", "factors_l3", "stats_l3"),
        ("Capa 4", "factors_l4", "stats_l4"),
    ]:
        facts = learning.get(fkey, {}); st = learning.get(skey, {})
        reliable = [(k, facts[k], learning.get(fkey.replace("factors","biases"), {}).get(k, 0),
                     st.get(k, {}).get("predicted", 0), st.get(k, {}).get("hits", 0))
                    for k in facts if st.get(k, {}).get("predicted", 0) >= 5]
        if not reliable:
            continue
        top = sorted(reliable, key=lambda x: x[1], reverse=True)[:5]
        print()
        print(f"  Top 5 {layer_name} (lift alto):")
        for k, lift, bias, pred, hits in top:
            prec = hits / pred if pred else 0
            print(f"    {k}  lift={lift:.3f}  bias={bias:+.3f}  "
                  f"prec={prec:.1%}  ({hits}/{pred})")
    print("=" * 62)
    print()


def cmd_history(args):
    sorteo   = args.sorteo.upper()
    learning = load_learning(sorteo)
    s_name   = "Nocturna" if sorteo == "N" else "Vespertina"
    history  = learning["history"][-args.last:]
    print()
    print(f"  HISTORIAL v1.2 {s_name} — últimas {len(history)} entradas")
    print(f"  {'Fecha':<12}  C1(E4)  C2(E4)  C3(E4)  C4(E4)   Estado")
    print(f"  {'-'*12}  {'------  '*4}  ------")
    for h in history:
        if h.get("actual") and h.get("hits_per_group"):
            def _t(key):
                d = h.get(key, {})
                return str(d.get(f"elite_{NUM_GROUPS}", "?")) if d else "—"
            c1 = _t("hits_cumulative")
            c2 = _t("hits_cumulative_l2")
            c3 = _t("hits_cumulative_l3")
            c4 = _t("hits_cumulative_l4")
            estado = "✓"
        else:
            c1 = c2 = c3 = c4 = "—"
            estado = "⏳ pendiente"
        print(f"  {h['date']:<12}  {c1:>6}  {c2:>6}  {c3:>6}  {c4:>6}   {estado}")
    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de pronóstico v1.2 — Tombola Uruguay (predicción en 3 capas)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("predict")
    pp.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    pp.add_argument("--numbers", nargs=20, required=True, metavar="NUM")
    pp.add_argument("--date", default=None, metavar="YYYY-MM-DD")
    pp.add_argument("--dow", type=int, default=None, choices=range(7), metavar="0-6")
    pp.add_argument("--plain", action="store_true")
    pp.add_argument("--html", default=None, metavar="PATH.html",
                    help="Generar archivo HTML interactivo con la predicción")

    pf = sub.add_parser("feedback")
    pf.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    pf.add_argument("--actual", nargs=20, required=True, metavar="NUM")
    pf.add_argument("--date", default=None)

    pa = sub.add_parser("accuracy")
    pa.add_argument("--sorteo", default="N", choices=["N","V","n","v"])

    ph = sub.add_parser("history")
    ph.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    ph.add_argument("--last", type=int, default=10)

    args = parser.parse_args()
    {"predict": cmd_predict, "feedback": cmd_feedback,
     "accuracy": cmd_accuracy, "history": cmd_history}[args.command](args)


if __name__ == "__main__":
    main()
