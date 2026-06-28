"""
tombola_predict_v11.py  ·  Versión 1.1
────────────────────────────────────────
Mejoras sobre v1.0:
  1. SCORING MULTI-VENTANA: blendea 4 ventanas (30d/90d/365d/all) con pesos
     configurables en lugar del blend fijo 60/40 histórico/reciente.
  2. AJUSTE DÍA-DE-SEMANA: aplica el dow_bias del JSON v11 para amplificar
     números que estadísticamente rinden más el día actual.
  3. SEÑAL COLD/CARRYOVER: números que llevan muchos sorteos sin aparecer reciben
     un pequeño boost ("están por salir"). Números que salieron ayer reciben un
     boost de carryover: la tasa real de repetición es ≈30% vs 20% base rate (1.5x).
  4. LIFT CON DECAY TEMPORAL: en lugar de tratar todos los registros del
     historial con igual peso, las rondas recientes pesan exponencialmente más.
     Esto hace que el modelo se adapte más rápido a cambios de patrón.
  5. UMBRAL DE CONFIANZA: números con score < SCORE_THRESHOLD se marcan como
     débiles (⚠). Solo E1+E2 se recomiendan para apostar; E3+E4 son referencia.
  6. GRUPOS CALCULADOS: la predicción muestra directamente los grupos verticales
     y combinados con identificación del eslabón débil de cada grupo.

COMANDOS
────────
1. Predecir — pasar la FECHA DEL SORTEO A PREDECIR con --date.
   El DOW se deriva automáticamente de esa fecha (no del día actual del sistema).
   Se puede ejecutar cualquier día de la semana para cualquier sorteo futuro.

   python tombola_predict_v11.py predict --sorteo N --date 2026-07-01 \\
     --numbers 04 08 11 18 30 48 49 50 52 54 60 65 67 68 72 74 80 87 90 94

   Si no se pasa --date, el DOW queda en None (sin ajuste de día).
   Para forzar un DOW diferente al de --date, usar --dow explícito (override).

2. Registrar resultado real:
   python tombola_predict_v11.py feedback --sorteo N --date 2026-05-23 \\
     --actual 00 06 07 08 10 16 17 19 27 28 31 40 46 49 56 61 74 80 90 98

3. Estadísticas:
   python tombola_predict_v11.py accuracy --sorteo N

4. Historial:
   python tombola_predict_v11.py history --sorteo N --last 5

REQUIERE
────────
tombola_N_transitions_v11.json (generado por tombola_analysis_v11.py)
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

# ── Pesos de ventana para el scoring ──────────────────────────────────────────
# Se suman hasta 1.0. Si una ventana tiene pocos datos, su peso se redistribuye
# automáticamente a next_day_all.
# next_day_5s: últimos 5 sorteos — captura clusters de muy corto plazo.
WINDOW_WEIGHTS = {
    "next_day_5s":   0.08,
    "next_day_30d":  0.09,
    "next_day_90d":  0.22,
    "next_day_365d": 0.32,
    "next_day_all":  0.29,
}

# ── Ajuste DOW ────────────────────────────────────────────────────────────────
DOW_INFLUENCE = 0.12   # cuánto influye el sesgo de día (0=ignorar, 1=máximo)

# ── Señal Cold/Carryover/Skip-day ─────────────────────────────────────────────
# Lifts empíricos medidos sobre 5750 pares históricos (Nocturna 2007-2026):
#   dsl=0 (carryover): P(aparecer)=20.10%  lift=1.005  (base rate=20%)
#   dsl=1 (skip-day):  P(aparecer)=20.04%  lift=1.002  (ruido estadístico)
#   dsl>=25 (frío):    P(aparecer)=20.19%  lift=1.010
# Los boosts reflejan los lifts reales; valores anteriores (10-12%) eran ~20x demasiado altos.
COLD_THRESHOLD   = 25    # sorteos sin aparecer = "frío"
COLD_BOOST       = 0.010 # lift empírico: 1.0097  (era 0.10 — 10x sobreestimado)
CARRYOVER_BOOST  = 0.020 # base 2% × lift individual del número (era 0.005 — insuficiente para carryovers con lift alto)
SKIPDAY_BOOST    = 0.010 # reintroducido: pequeño empuje para dsl=1 (era 0.000)

# ── Concentración de decil en input ───────────────────────────────────────────
# Si el sorteo anterior tuvo >= THR números del mismo decil, los candidatos
# de ese decil reciben un boost de momentum proporcional al exceso.
# Umbral=4 = 2x el promedio esperado (20 nums / 10 deciles = 2 por decil).
# Empírico (5750 pares): lift ~0.98x para count=5-6, ~1.20x para count>=7.
# Influencia conservadora para no amplificar el ruido en los casos intermedios.
INPUT_DECIL_CONC_THR       = 4    # mínimo para activar la señal (2x esperado)
INPUT_DECIL_CONC_INFLUENCE = 0.06 # boost por unidad de exceso sobre el umbral

# ── Grupos de apuesta ─────────────────────────────────────────────────────────
MAX_PER_DECIL  = 2         # máximo de números del mismo decil (00-09, 10-19…) por grupo
MIN_CARRYOVER_SLOTS = 3    # mínimo de carryovers garantizados en E1+E2 (de 12 posiciones)
MIN_HIGHLIFT_SLOTS  = 2    # mínimo de números con lift alto garantizados en E2
HIGHLIFT_THR        = 1.25 # lift mínimo para calificar (señal histórica confiable)
HIGHLIFT_MIN_PRED   = 20   # predicciones mínimas (evita overfitting de muestras pequeñas)
NUM_BETTING_GROUPS = 2     # E1+E2 para apostar; E3+E4 solo referencia
CANDIDATES_EXTRA   = 16    # números adicionales visibles más allá de E1-E4 (posiciones 25-40)
SCORE_THRESHOLD    = 1.05  # score mínimo para considerar un número confiable
SCORE_HIGH         = 1.50  # score de alta confianza (★)

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
DECAY       = 2.0       # tasa de decay temporal (mayor = más peso en datos recientes)

# ── Factores DOW-específicos ──────────────────────────────────────────────────
MIN_DOW_ROUNDS = 30    # mínimo de rondas por día para aprender factores DOW-específicos

# ── Ventana 5s adaptativa ─────────────────────────────────────────────────────
ADAPTIVE_TOP_N     = 15    # candidatos a comparar para coherencia 5s vs 30d
ADAPTIVE_THRESHOLD = 0.40  # overlap mínimo para mantener el peso de 5s

# ── Grupos por co-ocurrencia ──────────────────────────────────────────────────
COOCCUR_TOP_N      = 16    # candidatos a considerar para armar grupos
COOCCUR_INFLUENCE  = 0.30  # peso del lift de co-ocurrencia en joint_score
NUM_COOCCUR_GROUPS = 3     # grupos de apuesta a formar
COOCCUR_GROUP_SIZE = 4     # números por grupo


# ──────────────────────────────────────────────
# Rutas
# ──────────────────────────────────────────────

def transitions_path(sorteo: str) -> str:
    return os.path.join(BASE_DIR, f"tombola_{sorteo.upper()}_transitions_v11.json")

def learning_path(sorteo: str) -> str:
    return os.path.join(BASE_DIR, f"tombola_{sorteo.upper()}_learning_v11.json")


# ──────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────

def load_transitions(sorteo: str) -> dict:
    path = transitions_path(sorteo)
    if not os.path.exists(path):
        print(f"ERROR: No se encontró {path}")
        print(f"Ejecuta primero: python tombola_analysis_v11.py --sorteo {sorteo}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_learning(sorteo: str) -> dict:
    path = learning_path(sorteo)
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        # Migración si falta algún campo
        for field, default in [
            ("factors",     {f"{n:02d}": 1.0 for n in range(100)}),
            ("biases",      {f"{n:02d}": 0.0 for n in range(100)}),
            ("stats",       {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0, "w_predicted": 0.0, "w_hits": 0.0}
                             for n in range(100)}),
            ("dow_factors", {str(d): {f"{n:02d}": 1.0 for n in range(100)} for d in range(7)}),
            ("dow_biases",  {str(d): {f"{n:02d}": 0.0 for n in range(100)} for d in range(7)}),
        ]:
            if field not in data:
                data[field] = default
        return data
    return {
        "factors":     {f"{n:02d}": 1.0 for n in range(100)},
        "biases":      {f"{n:02d}": 0.0 for n in range(100)},
        "stats":       {f"{n:02d}": {"predicted": 0, "appeared": 0, "hits": 0,
                                     "w_predicted": 0.0, "w_hits": 0.0}
                        for n in range(100)},
        "dow_factors": {str(d): {f"{n:02d}": 1.0 for n in range(100)} for d in range(7)},
        "dow_biases":  {str(d): {f"{n:02d}": 0.0 for n in range(100)} for d in range(7)},
        "history":     [],
    }


def save_learning(sorteo: str, data: dict):
    with open(learning_path(sorteo), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _top_candidates(input_numbers: list, transitions: dict,
                    window_key: str, top_n: int) -> set:
    """Top-n candidatos de una ventana específica, ignorando el input."""
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
    """Lift bidireccional promedio entre a y b."""
    lift_ab = next((l for n, l, _ in cooccur.get(a, []) if n == b), 1.0)
    lift_ba = next((l for n, l, _ in cooccur.get(b, []) if n == a), 1.0)
    return (lift_ab + lift_ba) / 2.0


def _group_joint_score(nums: tuple, score_map: dict, cooccur: dict) -> float:
    """
    joint_score = sum(scores) × (1 + (avg_lift_pares - 1) × COOCCUR_INFLUENCE)

    Premia grupos donde los números tienden a salir juntos en el mismo sorteo.
    Si el lift promedio es 1.0 (independencia), el score es igual al individual.
    """
    ind_sum  = sum(score_map.get(n, 0.0) for n in nums)
    pairs    = list(combinations(nums, 2))
    avg_lift = sum(_pairwise_lift(a, b, cooccur) for a, b in pairs) / max(1, len(pairs))
    boost    = max(0.0, avg_lift - 1.0) * COOCCUR_INFLUENCE
    return ind_sum * (1.0 + boost)


def build_groups_cooccur(ranked: list, cooccur: dict) -> list:
    """
    Forma NUM_COOCCUR_GROUPS grupos de COOCCUR_GROUP_SIZE números desde los
    top COOCCUR_TOP_N candidatos, maximizando joint_score por co-ocurrencia.

    Usa selección greedy: evalúa C(16,4)=1820 combinaciones y elige las
    3 mejores no solapadas.

    Retorna: lista de (joint_score, avg_lift, grupo_tuple)
    """
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
# Scoring v1.1
# ──────────────────────────────────────────────

def compute_scores_v11(input_numbers: list[str],
                       transitions: dict,
                       factors: dict,
                       biases: dict,
                       dow: int | None = None) -> dict[str, float]:
    """
    Pipeline de scoring v1.1:

    1. Acumula probabilidades de transición blendando 4 ventanas temporales.
       Las ventanas con < 30 sorteos tienen su peso redistribuido a "all".

    2. Aplica ajuste DOW (día de la semana) si se proporciona.

    3. Aplica señal Cold/Carryover/Skip-day según days_since_last del JSON.

    4. Normaliza por RANGO [0,1] para amplificar diferencias.

    5. Aplica lift (multiplicativo) + bias (aditivo) del aprendizaje.
    """
    # ── Paso 1: Blend multi-ventana ─────────────────────────────────────────
    # Determinar qué ventanas tienen datos suficientes chequeando los
    # números de entrada reales (más robusto que chequear "00" fijo).
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

    # Redistribuir peso de ventanas sin datos a "next_day_all"
    if "next_day_all" in active_weights and len(active_weights) < len(WINDOW_WEIGHTS):
        missing = sum(v for k, v in WINDOW_WEIGHTS.items() if k not in active_weights)
        active_weights["next_day_all"] = active_weights.get("next_day_all", 0) + missing

    # ── Adaptativo: si next_day_5s diverge de next_day_30d → es ruido, redistribuir ──
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
                    continue  # p(i→i) ya está al base rate; excluirla evita auto-votar el input
                raw[candidate] += pct * (w_val / total_w)

    # Normalizar por cantidad de entradas
    n = len(input_numbers)
    base = {f"{i:02d}": raw.get(f"{i:02d}", 0.0) / n for i in range(100)}

    # ── Paso 2: Ajuste DOW ──────────────────────────────────────────────────
    if dow is not None:
        for k in base:
            dow_vec = transitions["transitions"].get(k, {}).get("dow_bias", [1.0] * 7)
            bias_d  = dow_vec[dow] if dow < len(dow_vec) else 1.0
            # Aplicar influencia controlada: no queremos mover demasiado el score
            base[k] *= (1.0 + (bias_d - 1.0) * DOW_INFLUENCE)

    # ── Paso 3: Señal Cold/Carryover/Skip-day ───────────────────────────────
    # Override dinámico: números en input_numbers aparecieron en el sorteo
    # anterior por definición (dsl_real=0), independiente del JSON que puede
    # estar desactualizado respecto al sorteo actual.
    input_set = set(input_numbers)
    for k in base:
        if k in input_set:
            effective_dsl = 0
        else:
            dsl_json = transitions["transitions"].get(k, {}).get("days_since_last")
            if dsl_json is None:
                continue
            # Si el JSON dice dsl=0 pero el número NO está en el input, el JSON está
            # desactualizado: ese número apareció en el pre-input (N-2), no en el
            # input (N-1). Su dsl real es 1 (skip-day), no 0 (carryover).
            effective_dsl = 1 if dsl_json == 0 else dsl_json
        if effective_dsl >= COLD_THRESHOLD:
            factor = min(effective_dsl / (COLD_THRESHOLD * 3), 1.0)
            base[k] *= (1.0 + COLD_BOOST * factor)
        elif effective_dsl == 0:
            # Carryover: boost base × lift individual del número.
            lift_factor = factors.get(k, 1.0)
            base[k] *= (1.0 + CARRYOVER_BOOST * lift_factor)
        elif effective_dsl == 1:
            base[k] *= (1.0 + SKIPDAY_BOOST)

    # ── Paso 3b: Ajuste de decil (frío/caliente) ────────────────────────────
    # decil_bias > 1.0 → decil frío, boost. < 1.0 → decil caliente, penalización.
    for k in base:
        db = transitions["transitions"].get(k, {}).get("decil_bias", 1.0)
        base[k] *= db

    # ── Paso 3c: Concentración de decil en el input ──────────────────────────
    # Calcula qué deciles están sobrerrepresentados en el sorteo anterior
    # y aplica un boost de momentum a los candidatos de esos deciles.
    # Se activa cuando un decil tiene >= INPUT_DECIL_CONC_THR números en input
    # (umbral = 2x el promedio esperado de 2 por decil).
    input_decil_cnt: dict[int, int] = {}
    for n in input_numbers:
        d = int(n) // 10
        input_decil_cnt[d] = input_decil_cnt.get(d, 0) + 1

    for k in base:
        d   = int(k) // 10
        cnt = input_decil_cnt.get(d, 0)
        if cnt >= INPUT_DECIL_CONC_THR:
            excess   = cnt - INPUT_DECIL_CONC_THR + 1  # 1 si cnt=4, 2 si cnt=5 …
            base[k] *= (1.0 + INPUT_DECIL_CONC_INFLUENCE * excess)

    # ── Paso 4: Blend rank [0,1] + score normalizado [0,1] ──────────────────
    # Rank puro exagera diferencias pequeñas en el tope y las aplana en el medio.
    # Mezclar con la magnitud real preserva separación estadísticamente significativa.
    sorted_nums = sorted(base, key=base.get, reverse=True)
    rank_score  = {num: 1.0 - (i / 99.0) for i, num in enumerate(sorted_nums)}

    min_s = min(base.values())
    max_s = max(base.values())
    rng   = max_s - min_s if max_s > min_s else 1.0
    norm_raw = {k: (v - min_s) / rng for k, v in base.items()}

    RANK_BLEND = 0.5
    blended = {k: RANK_BLEND * rank_score[k] + (1.0 - RANK_BLEND) * norm_raw[k]
               for k in base}

    # ── Paso 5: Lift × bias del aprendizaje ─────────────────────────────────
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
    """
    Forma los grupos con cap de diversidad por decil (MAX_PER_DECIL).
    Aplica dos garantías de slots en E2 (en orden):
      1. MIN_CARRYOVER_SLOTS: al menos N carryovers del input en E1+E2.
      2. MIN_HIGHLIFT_SLOTS: al menos N números con lift >= HIGHLIFT_THR
         y >= HIGHLIFT_MIN_PRED predicciones históricas en E1+E2.
    Cada garantía reemplaza los peores slots no-calificados de E2.
    """
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

    # ── Carryover mínimo garantizado en E1+E2 ────────────────────────────────
    if input_numbers and MIN_CARRYOVER_SLOTS > 0:
        input_set  = set(input_numbers)
        betting    = groups["elite_1"] + groups["elite_2"]
        cy_in      = sum(1 for n, _ in betting if n in input_set)
        slots_need = MIN_CARRYOVER_SLOTS - cy_in

        if slots_need > 0:
            # Excluir números ya usados en CUALQUIER grupo (E1–E4), no solo E1+E2
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

    # ── High-lift mínimo garantizado en E1+E2 ────────────────────────────────
    if stats is not None and MIN_HIGHLIFT_SLOTS > 0:
        betting     = groups["elite_1"] + groups["elite_2"]
        factors_ref = stats.get("__factors__", {})  # pasado desde el caller

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
# Aprendizaje v1.1: lift con decay temporal
# ──────────────────────────────────────────────

def recompute_learning_v11(history: list[dict]) -> tuple[dict, dict, dict]:
    """
    Recalcula factores y biases desde todo el historial con DECAY TEMPORAL.

    Rondas recientes pesan más (weight = exp(-DECAY × (1 - i/total))):
      - Ronda más reciente: weight = 1.00
      - Ronda de hace 100 rondas (de 500): weight ≈ 0.67
      - Ronda más antigua:  weight ≈ exp(-DECAY) ≈ 0.13

    Esto hace que el modelo se adapte más rápido a cambios de patrón recientes
    sin olvidar completamente el comportamiento histórico.
    """
    completed = [h for h in history
                 if h.get("actual") and h.get("predicted_groups")]
    total = len(completed)

    # Contadores (pesados temporalmente)
    w_predicted = defaultdict(float)
    w_hits      = defaultdict(float)
    appeared    = defaultdict(int)

    for i, entry in enumerate(completed):
        # weight: ronda más reciente → peso 1.0, más antigua → peso exp(-DECAY)
        w = math.exp(-DECAY * (total - 1 - i) / max(1, total - 1))

        actual_set = set(entry["actual"])
        for num in actual_set:
            appeared[num] += 1

        for nums in entry.get("predicted_groups", {}).values():
            for num in nums:
                w_predicted[num] += w
                if num in actual_set:
                    w_hits[num] += w

    # Base rate ponderada (≈ 0.20 pero puede variar si el CSV tiene días distintos)
    base_rate = BASE_RATE

    factors = {}
    biases  = {}
    stats   = {}

    for n in range(100):
        k   = f"{n:02d}"
        wp  = w_predicted.get(k, 0.0)
        wh  = w_hits.get(k, 0.0)
        raw_p = int(wp + 0.5)   # aproximación entera para compatibilidad con stats

        # Laplace smoothing sobre conteos ponderados
        precision = (wh + LAPLACE_K * base_rate) / (wp + LAPLACE_K)

        lift  = precision / base_rate
        lift  = round(max(MIN_FACTOR, min(MAX_FACTOR, lift)), 4)

        raw_bias = (precision - base_rate) * BIAS_SCALE
        bias  = round(max(MIN_BIAS, min(MAX_BIAS, raw_bias)), 4)

        factors[k] = lift
        biases[k]  = bias
        stats[k]   = {
            "predicted":   raw_p,
            "appeared":    appeared.get(k, 0),
            "hits":        int(wh + 0.5),
            "w_predicted": round(wp, 2),
            "w_hits":      round(wh, 2),
        }

    # ── DOW-específico: factores por día de la semana ─────────────────────────
    # Fallback: copiar globales para días con pocos datos
    dow_factors = {str(d): dict(factors) for d in range(7)}
    dow_biases  = {str(d): dict(biases)  for d in range(7)}

    by_dow = defaultdict(list)
    for entry in completed:
        d = entry.get("dow")
        if d is not None:
            by_dow[str(d)].append(entry)

    for d_str, d_entries in by_dow.items():
        if len(d_entries) < MIN_DOW_ROUNDS:
            continue  # pocos datos → mantener fallback global
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
            precision = (wh + LAPLACE_K * base_rate) / (wp + LAPLACE_K)
            lift = round(max(MIN_FACTOR, min(MAX_FACTOR, precision / base_rate)), 4)
            bias = round(max(MIN_BIAS, min(MAX_BIAS, (precision - base_rate) * BIAS_SCALE)), 4)
            dow_factors[d_str][k] = lift
            dow_biases[d_str][k]  = bias

    return factors, biases, stats, dow_factors, dow_biases


# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────

GROUP_LABELS = {
    "elite_1": ("🥇", "ÉLITE 1", "máxima confianza — APOSTAR"),
    "elite_2": ("🥈", "ÉLITE 2", "alta confianza — APOSTAR"),
    "elite_3": ("🔵", "REFERENCIA 3", "no apostar"),
    "elite_4": ("⚪", "REFERENCIA 4", "no apostar"),
}
DOW_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


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
    """Muestra grupos verticales y combinados con identificación del eslabón débil."""
    e1 = groups.get("elite_1", [])
    e2 = groups.get("elite_2", [])
    e3 = groups.get("elite_3", [])
    e4 = groups.get("elite_4", [])

    def tag(num: str, score: float, ref: bool = False) -> str:
        if ref or score < SCORE_THRESHOLD:
            return f"{num}⚠"
        if score >= SCORE_HIGH:
            return f"{num}★"
        return num

    print("  ── Grupos Calculados ─────────────────────────────────────────")
    print()
    print("  Verticales [E1 | E2 | E3* | E4*]  (*=referencia)")
    for i in range(GROUP_SIZE):
        row_tags = []
        for j, elite in enumerate([e1, e2, e3, e4]):
            if i < len(elite):
                n, s = elite[i]
                is_ref = j >= NUM_BETTING_GROUPS
                label  = f"{tag(n, s, ref=is_ref)}({s:.3f})"
                row_tags.append(label)
            else:
                row_tags.append("--")
        print(f"    V{i+1}: [{', '.join(row_tags)}]")

    print()
    print("  Combinados E1+E2 — APOSTAR:")
    for i in range(0, GROUP_SIZE, 2):
        row = []
        for elite in [e1, e2]:
            for j in [i, i + 1]:
                if j < len(elite):
                    row.append(elite[j])
        ci     = i // 2 + 1
        parts  = [
            f"{n}({s:.3f})" + ("⚠" if s < SCORE_THRESHOLD else ("★" if s >= SCORE_HIGH else ""))
            for n, s in row
        ]
        n_weak = sum(1 for _, s in row if s < SCORE_THRESHOLD)
        flag   = f"  [{n_weak} débil{'es' if n_weak > 1 else ''}]" if n_weak else ""
        print(f"    C{ci}: [{', '.join(parts)}]{flag}")

    print()
    print("  Combinados E3+E4 — referencia:")
    for i in range(0, GROUP_SIZE, 2):
        row = []
        for elite in [e3, e4]:
            for j in [i, i + 1]:
                if j < len(elite):
                    row.append(elite[j])
        ci     = i // 2 + 4
        parts  = [
            f"{n}({s:.3f})" + ("⚠" if s < SCORE_THRESHOLD else ("★" if s >= SCORE_HIGH else ""))
            for n, s in row
        ]
        n_weak = sum(1 for _, s in row if s < SCORE_THRESHOLD)
        flag   = f"  [{n_weak} débil{'es' if n_weak > 1 else ''}]" if n_weak else ""
        print(f"    C{ci}: [{', '.join(parts)}]{flag}")

    print()
    print(f"  Leyenda: ★ score ≥ {SCORE_HIGH}  ⚠ score < {SCORE_THRESHOLD} (débil/referencia)")


def print_cooccur_groups(cooccur_groups: list, score_map: dict, cooccur: dict) -> None:
    """Muestra los grupos formados por co-ocurrencia con sus pares internos."""
    print()
    print("  " + "=" * 58)
    print("  GRUPOS CO-OCURRENCIA — objetivo: predecir los 4 del grupo")
    print("  " + "=" * 58)
    labels = ["CG1 (principal)", "CG2", "CG3"]
    for i, (js, avg_lift, grp) in enumerate(cooccur_groups):
        nums_str = "  ".join(grp)
        scores   = [score_map.get(n, 0.0) for n in grp]
        tag      = "  ★ APOSTAR" if i == 0 else ""
        print(f"  {labels[i]}: {nums_str}   lift_avg={avg_lift:.3f}{tag}")

        # Pares internos ordenados por lift mutuo
        pairs = sorted(combinations(grp, 2),
                       key=lambda ab: _pairwise_lift(ab[0], ab[1], cooccur),
                       reverse=True)
        pair_strs = [f"[{a}-{b}]({_pairwise_lift(a,b,cooccur):.2f}x)" for a, b in pairs[:3]]
        print(f"           pares: {' '.join(pair_strs)}")
    print()
    print("  Para registrar 4 aciertos exactos en un grupo:")
    print("  Aposta CG1 completo como unidad (los 4 números).")
    print("  " + "=" * 58)


def print_prediction(sorteo, input_numbers, groups, learning, dow, show_scores,
                     cooccur_groups=None, score_map=None, cooccur=None):
    s_name = "Nocturna" if sorteo == "N" else "Vespertina"
    rondas = len(learning["history"])
    dow_txt = (f"  Día del sorteo: {DOW_NAMES[dow]} (DOW={dow})" if dow is not None
               else "  Día del sorteo: no especificado — sin ajuste DOW (pasar --date YYYY-MM-DD)")

    print()
    print("=" * 62)
    print(f"  PRONÓSTICO v1.1 — TOMBOLA {s_name.upper()} ({sorteo})")
    print("=" * 62)
    print(f"  Entrada ({len(input_numbers)} números): {' '.join(sorted(input_numbers))}")
    if dow_txt:
        print(dow_txt)
    print(f"  Rondas aprendidas: {rondas}")
    print()

    # ── Grupos de apuesta (E1 y E2) ──────────────────────────────────────────
    for i in range(NUM_BETTING_GROUPS):
        key    = f"elite_{i + 1}"
        items  = groups.get(key, [])
        emoji, label, desc = GROUP_LABELS.get(key, ("⚪", key.upper(), ""))
        print(f"  {emoji} {label}  ({GROUP_SIZE} números — {desc})")
        if show_scores:
            print(fmt_group(items, cols=6, threshold=SCORE_THRESHOLD))
        else:
            print(fmt_plain(items, cols=6))
        print()

    # ── Grupos de referencia (E3 y E4) ───────────────────────────────────────
    print(f"  {'─' * 58}")
    for i in range(NUM_BETTING_GROUPS, NUM_GROUPS):
        key    = f"elite_{i + 1}"
        items  = groups.get(key, [])
        emoji, label, desc = GROUP_LABELS.get(key, ("⚪", key.upper(), ""))
        print(f"  {emoji} {label}  ({GROUP_SIZE} números — {desc})")
        if show_scores:
            print(fmt_group(items, cols=6))
        else:
            print(fmt_plain(items, cols=6))
        print()

    # ── Candidatos extra: primeros CANDIDATES_EXTRA de RESTO (posiciones 25-40) ──
    resto_all  = groups.get("resto", [])
    extra      = resto_all[:CANDIDATES_EXTRA]
    resto_tail = resto_all[CANDIDATES_EXTRA:]
    print(f"  {'─' * 58}")
    print(f"  📋 CANDIDATOS EXTRA ({len(extra)} números — visibilidad, no apostar)")
    if show_scores:
        print(fmt_group(extra, cols=6, threshold=0.0))
    else:
        print(fmt_plain(extra, cols=8))
    print()

    resto_nums = " ".join(n for n, _ in resto_tail)
    print(f"  ⚪ RESTO  ({len(resto_tail)} números — zona no predicha)")
    print(f"  {resto_nums}")
    print()

    print_betting_groups(groups)

    if cooccur_groups and score_map is not None and cooccur is not None:
        print_cooccur_groups(cooccur_groups, score_map, cooccur)

    print()
    print("  Para registrar el resultado:")
    print(f'  python tombola_predict_v11.py feedback --sorteo {sorteo} \\')
    print(f'    --date {date.today().isoformat()} --actual <20 numeros>')
    print("=" * 62)
    print()


# ──────────────────────────────────────────────
# Comandos
# ──────────────────────────────────────────────

def cmd_predict(args):
    sorteo   = args.sorteo.upper()
    trans    = load_transitions(sorteo)
    learning = load_learning(sorteo)

    # Resolver DOW: --dow explícito > derivado de --date > None (sin ajuste)
    # Nunca se usa la fecha del sistema como base para el DOW.
    if args.dow is not None:
        dow = args.dow
    elif args.date is not None:
        dow = datetime.strptime(args.date, "%Y-%m-%d").weekday()
    else:
        dow = None
        print("  AVISO: sin --date no se puede determinar el día del sorteo.")
        print("         Usá --date YYYY-MM-DD con la fecha del sorteo a predecir.")
        print("         Ejemplo: --date 2026-06-23  (el script corre cualquier día)")
        print()

    raw = args.numbers
    if len(raw) != 20:
        print(f"ERROR: Se necesitan 20 números (recibidos: {len(raw)})."); sys.exit(1)
    try:
        input_numbers = [f"{int(n):02d}" for n in raw]
    except ValueError:
        print("ERROR: Números inválidos."); sys.exit(1)

    # Usar factores DOW-específicos si están disponibles
    if dow is not None and str(dow) in learning.get("dow_factors", {}):
        eff_factors = learning["dow_factors"][str(dow)]
        eff_biases  = learning["dow_biases"].get(str(dow), learning["biases"])
    else:
        eff_factors = learning["factors"]
        eff_biases  = learning["biases"]
    scores  = compute_scores_v11(input_numbers, trans, eff_factors, eff_biases, dow)
    ranked  = rank_candidates(scores)
    hl_stats = dict(learning.get("stats", {}))
    hl_stats["__factors__"] = learning["factors"]  # factors globales (no DOW-específicos)
    groups  = build_groups(ranked, input_numbers, hl_stats)

    cooccur        = trans.get("cooccur", {})
    cooccur_groups = build_groups_cooccur(ranked, cooccur) if cooccur else None
    score_map      = {n: s for n, s in ranked}

    print_prediction(sorteo, input_numbers, groups, learning, dow,
                     show_scores=not args.plain,
                     cooccur_groups=cooccur_groups,
                     score_map=score_map,
                     cooccur=cooccur)

    predicted_groups = {
        f"elite_{i+1}": [n for n, _ in groups.get(f"elite_{i+1}", [])]
        for i in range(NUM_GROUPS)
    }
    pending = {
        "date":             args.date or date.today().isoformat(),
        "sorteo":           sorteo,
        "dow":              dow,
        "input":            input_numbers,
        "predicted_groups": predicted_groups,
        "actual":           None,
        "hits_per_group":   None,
        "hits_cumulative":  None,
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
    print(f"  ✓ Predicción guardada. Registra el resultado con `feedback`.")
    print()


def cmd_feedback(args):
    sorteo  = args.sorteo.upper()
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
    pred_groups = entry.get("predicted_groups", {})

    hits_per_group  = {}
    hits_cumulative = {}
    cumset = set()
    for i in range(NUM_GROUPS):
        gk  = f"elite_{i+1}"
        gs  = set(pred_groups.get(gk, []))
        hits_per_group[gk]  = len(actual_set & gs)
        cumset |= gs
        hits_cumulative[gk] = len(actual_set & cumset)

    entry["actual"]          = actual
    entry["hits_per_group"]  = hits_per_group
    entry["hits_cumulative"] = hits_cumulative

    new_factors, new_biases, new_stats, new_dow_f, new_dow_b = recompute_learning_v11(history)
    learning["factors"]     = new_factors
    learning["biases"]      = new_biases
    learning["stats"]       = new_stats
    learning["dow_factors"] = new_dow_f
    learning["dow_biases"]  = new_dow_b
    history[idx]            = entry
    save_learning(sorteo, learning)

    s_name = "Nocturna" if sorteo == "N" else "Vespertina"
    print()
    print("=" * 56)
    print(f"  RESULTADO REGISTRADO — {s_name} {target_date}")
    print("=" * 56)
    print(f"  Números reales: {' '.join(sorted(actual))}")
    print()
    print(f"  {'Grupo':<10} {'Predichos':>9} {'Aciertos':>9} {'Acum.':>7} {'Azar':>6}")
    print(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*7} {'-'*6}")
    ac = 0
    for i in range(NUM_GROUPS):
        gk   = f"elite_{i+1}"
        h    = hits_per_group[gk]
        hc   = hits_cumulative[gk]
        ac   += GROUP_SIZE
        azar = round(ac / 100 * 20, 1)
        print(f"  {gk:<10} {GROUP_SIZE:>9} {h:>7}/20  {hc:>5}/20  {azar:>5}")
    all_pred = set(n for g in pred_groups.values() for n in g)
    missed   = sorted(actual_set - all_pred)
    if missed:
        print(f"\n  No predichos: {' '.join(missed)}")
    print(f"\n  ✓ Factores actualizados con decay temporal (DECAY={DECAY}).")
    print("=" * 56)
    print()


def cmd_accuracy(args):
    sorteo  = args.sorteo.upper()
    learning = load_learning(sorteo)
    s_name  = "Nocturna" if sorteo == "N" else "Vespertina"
    completed = [h for h in learning["history"]
                 if h.get("actual") and h.get("hits_per_group")]
    n = len(completed)
    if n == 0:
        print(f"Sin rondas completadas para {s_name}."); return

    print()
    print("=" * 60)
    print(f"  PRECISIÓN v1.1 — {s_name} ({sorteo})")
    print("=" * 60)
    print(f"  Rondas evaluadas: {n}  |  Decay temporal: {DECAY}")
    print()
    print(f"  {'Grupo':<10} {'Prom.solo':>10} {'Prom.acum':>10} {'Azar':>6}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for i in range(NUM_GROUPS):
        gk   = f"elite_{i+1}"
        solo = sum(h["hits_per_group"].get(gk, 0) for h in completed) / n
        acum = sum(h.get("hits_cumulative", {}).get(gk, 0) for h in completed) / n
        azar = round(GROUP_SIZE * (i + 1) / 100 * 20, 1)
        print(f"  {gk:<10} {solo:>9.2f}  {acum:>9.2f}  {azar:>6.1f}")

    # Últimas rondas
    last  = completed[-5:]
    hdr   = "  ".join(f"E{i+1}" for i in range(NUM_GROUPS))
    print()
    print(f"  Últimas {len(last)} rondas:")
    print(f"  {'Fecha':<12}  {hdr}  | total")
    for h in last:
        hpg   = h.get("hits_per_group", {})
        hcm   = h.get("hits_cumulative", {})
        cols  = "   ".join(str(hpg.get(f"elite_{i+1}", "-")) for i in range(NUM_GROUPS))
        total = hcm.get(f"elite_{NUM_GROUPS}", "-")
        print(f"  {h['date']:<12}  {cols}  | {total}/20")

    # Factores top
    stats = learning.get("stats", {})
    reliable = [(k, learning["factors"][k], learning["biases"].get(k, 0),
                 stats.get(k, {}).get("predicted", 0),
                 stats.get(k, {}).get("hits", 0))
                for k in learning["factors"]
                if stats.get(k, {}).get("predicted", 0) >= 5]
    if reliable:
        top = sorted(reliable, key=lambda x: x[1], reverse=True)[:8]
        low = sorted(reliable, key=lambda x: x[1])[:5]
        print()
        print("  Top 8 números reforzados (lift alto — decay-weighted):")
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
    print("=" * 60)
    print()


def cmd_history(args):
    sorteo  = args.sorteo.upper()
    learning = load_learning(sorteo)
    s_name  = "Nocturna" if sorteo == "N" else "Vespertina"
    history = learning["history"][-args.last:]
    hdr     = "  ".join(f"E{i+1}" for i in range(NUM_GROUPS))
    print()
    print(f"  HISTORIAL v1.1 {s_name} — últimas {len(history)} entradas")
    print(f"  {'Fecha':<12}  {hdr}  | total   Estado")
    print(f"  {'-'*12}  {'  '.join(['--']*NUM_GROUPS)}  | ----   ------")
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
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de pronóstico v1.1 — Tombola Uruguay (multi-ventana + DOW + decay)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # predict
    pp = sub.add_parser("predict")
    pp.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    pp.add_argument("--numbers", nargs=20, required=True, metavar="NUM")
    pp.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                    help="Fecha del sorteo a predecir. El DOW se deriva automáticamente "
                         "de esta fecha, sin importar cuándo se ejecute el script.")
    pp.add_argument("--dow", type=int, default=None, choices=range(7),
                    metavar="0-6",
                    help="Override manual del día de la semana (0=Lun … 6=Dom). "
                         "Solo necesario si querés forzar un DOW distinto al de --date.")
    pp.add_argument("--plain", action="store_true")

    # feedback
    pf = sub.add_parser("feedback")
    pf.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    pf.add_argument("--actual", nargs=20, required=True, metavar="NUM")
    pf.add_argument("--date", default=None)

    # accuracy
    pa = sub.add_parser("accuracy")
    pa.add_argument("--sorteo", default="N", choices=["N","V","n","v"])

    # history
    ph = sub.add_parser("history")
    ph.add_argument("--sorteo", default="N", choices=["N","V","n","v"])
    ph.add_argument("--last", type=int, default=10)

    args = parser.parse_args()
    {"predict": cmd_predict, "feedback": cmd_feedback,
     "accuracy": cmd_accuracy, "history": cmd_history}[args.command](args)


if __name__ == "__main__":
    main()
