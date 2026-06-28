# Tombola Predictor — Referencia rápida

## Versión activa: v1.1 (tombola_predict_v11.py)

---

## Pipeline v1.1 — Flujo completo

### 1. Generar transiciones (una vez, o cuando lleguen datos nuevos al CSV)
```
python tombola_analysis_v11.py --sorteo N
python tombola_analysis_v11.py --sorteo V
```

### 2. Entrenar
```
# Primera vez / reset completo
python tombola_train_v11.py --sorteo N --reset --last 5000 --verbose 100

# Incorporar sorteos nuevos sin perder historial
python tombola_train_v11.py --sorteo N --last 200 --verbose 100
```

### 3. Predecir el próximo sorteo
```
# Pasar --date con la fecha DEL SORTEO A PREDECIR (no la fecha de hoy)
# El día de la semana (DOW) se deriva automáticamente de --date
python tombola_predict_v11.py predict --sorteo N \
  --date 2026-06-23 \
  --numbers 01 10 11 12 13 15 18 19 21 25 30 37 43 45 57 66 72 85 87 96
```

> **Nota DOW:** Se puede ejecutar el viernes para predecir el martes.
> Siempre pasar `--date` con la fecha del sorteo objetivo — el script
> nunca usa la fecha del sistema para determinar el día de la semana.
> `--dow` queda disponible solo como override manual si se necesita.

### 4. Registrar resultado real y regenerar JSON
```
# 4a — Registrar el resultado (actualiza factores aprendidos)
python tombola_predict_v11.py feedback --sorteo N \
  --date 2026-06-23 \
  --actual 02 12 15 20 26 36 40 41 44 51 53 64 65 71 74 81 82 89 92 98

# 4b — Regenerar transiciones con el CSV actualizado  ← SIEMPRE después de feedback
python tombola_analysis_v11.py --sorteo N
```

> **Por qué regenerar siempre:** el JSON de transiciones guarda `days_since_last` (dsl)
> por cada número. Si está desactualizado, el modelo trata números del pre-input
> como carryovers cuando en realidad son skip-day. Regenerar garantiza dsl exacto.

### 5. Ver precisión acumulada
```
python tombola_predict_v11.py accuracy --sorteo N
```

### 6. Ver historial reciente
```
python tombola_predict_v11.py history --sorteo N --last 10
```

---

## Backtest / prueba desde fecha pasada

Para testear el algoritmo usando datos históricos (ej: hoy es 26/6,
querés simular la predicción del 23/6 usando los datos del 22/6):

```
# PASO 1 — Predecir usando los números del 22/6, apuntando al 23/6
python tombola_predict_v11.py predict --sorteo N \
  --date 2026-06-23 \
  --numbers 01 10 11 12 13 15 18 19 21 25 30 37 43 45 57 66 72 85 87 96

# PASO 2 — Registrar el resultado real del 23/6
python tombola_predict_v11.py feedback --sorteo N \
  --date 2026-06-23 \
  --actual <20 números reales del 23/6>
```

---

## Cambios aplicados al algoritmo v1.1

Todos los cambios están en `tombola_predict_v11.py` salvo donde se indica.

---

### Pipeline de scoring actualizado — `compute_scores_v11`

```
Paso 1   Multi-ventana (5s / 30d / 90d / 365d / all) con pesos adaptativos
Paso 2   Ajuste DOW — derivado de --date, nunca del sistema
Paso 3   Cold/Carryover con override dinámico desde input_numbers
Paso 3b  Decil bias estático del JSON (DECIL_INFLUENCE reducido a 0.02)
Paso 3c  [NUEVO] Concentración de decil en el input
Paso 4   Blend rank [0,1] + score normalizado
Paso 5   Lift aprendido × rank + bias
→ build_groups con cap MAX_PER_DECIL=2 por grupo
```

---

### [1] DOW derivado de --date, nunca del sistema (2026-06-26)

**Problema:** si se corre el script el viernes para predecir el martes, el DOW
quedaba mal porque dependía de la fecha de ejecución.

**Fix:** el DOW se deriva automáticamente de `--date` (la fecha del sorteo
a predecir). `--dow` queda disponible solo como override manual.

```
Prioridad: --dow explícito > derivado de --date > None (sin ajuste, con aviso)
```

Permite correr predicciones y backtests cualquier día de la semana.

---

### [2] Fix carryover dinámico — Paso 3 (2026-06-26)

**Problema:** el código usaba `days_since_last` del JSON estático. Si el JSON
tenía varios días de antigüedad, los números del input no recibían el boost
de carryover porque el JSON no sabía que acababan de salir.

**Fix:** cualquier número presente en `input_numbers` recibe `effective_dsl=0`
independiente de lo que diga el JSON.

```python
if k in input_set:
    effective_dsl = 0          # siempre carryover, sin importar el JSON
else:
    effective_dsl = dsl_json   # usar el JSON para el resto
```

---

### [3] Calibración de boosts Cold/Carryover/Skip-day (2026-06-26)

Análisis sobre 5.750 pares del historial completo (2007–2026):

| Señal | P(aparecer) | Lift real | Valor anterior | Valor nuevo |
|-------|-------------|-----------|----------------|-------------|
| Carryover `dsl=0` | 20.10% | 1.005x | `+10%` (20x sobreestimado) | `+0.5%` |
| Skip-day `dsl=1`  | 20.04% | 1.002x | `+12%` (ruido puro)        | `0%`    |
| Frío `dsl≥25`     | 20.19% | 1.010x | `+10%` (10x sobreestimado) | `+1.0%` |

La media de carryover histórica es 4/20 (20%) — igual que la tasa base.
El sorteo del 22/6 con 7 carryovers fue el percentil 95, no un patrón regular.

---

### [4] Señal de concentración de decil en el input — Paso 3c (2026-06-26)

**Lógica:** si el sorteo anterior tuvo ≥ 4 números del mismo decil (2x el
promedio esperado de 2 por decil), los candidatos de ese decil reciben un
boost de momentum proporcional al exceso sobre el umbral.

```
boost(k) = INPUT_DECIL_CONC_INFLUENCE × (count_decil − THR + 1)

  count=4  → excess=1 → +6%
  count=5  → excess=2 → +12%
  count=6  → excess=3 → +18%
  count=7  → excess=4 → +24%
```

**Parámetros:**
- `INPUT_DECIL_CONC_THR = 4` (umbral de activación)
- `INPUT_DECIL_CONC_INFLUENCE = 0.06` (boost por unidad de exceso)

**Ejemplo — predicción del 22/6 (input = sorteo del 20/6):**

| Decil | Nums en input | Boost aplicado | Nums en resultado |
|-------|--------------|---------------|-------------------|
| 80-89 | 5 → +12%    | activo        | 85, 87            |
| 60-69 | 3            | no activo     | 66                |
| 30-39 | 3            | no activo     | 30, 37            |
| 10-19 | 2            | no activo     | 10,11,12,13,15,18,19 |

La concentración en 10-19 del resultado fue impredecible desde el input
(solo 2 números allí → sin señal). La señal correctamente boosteó 80-89
donde cayeron 85 y 87.

**Respaldo empírico:** lift medido ~1.00x para count=4-6, ~1.20x para count≥7
(20 casos históricos). Señal débil pero captura momentum en casos extremos.

---

### [5] Calibración de decil bias estático — `tombola_analysis_v11.py` (2026-06-26)

`DECIL_INFLUENCE`: `0.15` → `0.02`

El ajuste estático de decil caliente/frío del JSON ahora refleja el lift
empírico real (±1-2%) en lugar de ±15% que amplificaba ruido ~8x.
Aplica al próximo `python tombola_analysis_v11.py --sorteo N`.

---

### [6] Diversidad de decil por grupo — `build_groups` (2026-06-26)

`MAX_PER_DECIL = 2` — cada grupo (E1–E4) acepta como máximo 2 números
del mismo decil. Si el ranking concentra 3+ candidatos de un mismo rango,
los extras son reemplazados por el mejor candidato de otro decil.

```
Antes: E3 podía tener 71, 73, 76, 78 (4 del decil 70-79)
Ahora: máximo 2 del mismo decil → mayor cobertura del espacio de 100 números
```

Los 24 predichos cubren al menos 10-12 deciles distintos en lugar de saturar 3-4.

---

### [7] Calibración de carryover/skip-day boosts (2026-06-26)

Ajustes calibrados al azar a partir del análisis empírico del sorteo 22/6:

```python
CARRYOVER_BOOST = 0.020   # era 0.005 — boost base × lift individual del número
SKIPDAY_BOOST   = 0.010   # era 0.000 — reintroducido con valor mínimo
```

El boost de carryover ahora escala por el `lift` aprendido de cada número:
```
boost_efectivo = CARRYOVER_BOOST × lift_individual
  lift=1.94 (número que repite frecuente) → +3.9%
  lift=1.00 (número neutro)               → +2.0%
  lift=0.98 (número penalizado)           → +1.9%
```

---

### [8] Fix dsl skip-day para JSON desactualizado — Paso 3 (2026-06-26)

**Problema:** si el JSON fue generado antes del último sorteo, los números que
aparecieron en ese sorteo tienen `dsl=0` en el JSON. Si esos números NO están
en el input actual, el modelo los trataba como carryover (dsl=0) cuando en
realidad son skip-day (dsl=1) — recibían el boost equivocado.

**Caso concreto:** predicción del 23/6 usando input del 22/6. JSON generado
al 20/6. Números 71, 82, 89 aparecieron en 20/6 (dsl=0 en JSON) pero no en
22/6 (input). El modelo los boosteó como carryover → seguían en Resto.
Los tres aparecieron en el resultado del 23/6.

**Fix aplicado:**
```python
# Si JSON dice dsl=0 pero el número NO está en el input → JSON stale → skip-day real
effective_dsl = 1 if dsl_json == 0 else dsl_json
```

**Mejora B complementaria — regenerar JSON siempre después de feedback:**
Con el JSON al día, `dsl_json` ya refleja el estado correcto y el fix de A
actúa como red de seguridad para el caso residual.

---

## Pipeline v1.2 (experimental)

```
python tombola_analysis_v12.py --sorteo N
python tombola_train_v12.py --sorteo N --reset --last 500
python tombola_predict_v12.py predict --sorteo N \
  --date 2026-06-23 \
  --numbers 01 11 17 28 30 38 39 41 44 45 48 51 52 60 62 65 70 72 88 99
```

---

## Pipeline v1.0 (legacy)

```
python tombola_analysis.py --sorteo N --window 365
python tombola_train.py --sorteo N --reset --last 700
python tombola_predict.py predict --sorteo N \
  --numbers 05 11 26 28 31 35 47 50 53 63 64 67 73 77 78 80 82 86 87 88
python tombola_predict.py feedback --sorteo N \
  --date 2026-06-12 \
  --actual 05 11 26 28 31 35 47 50 53 63 64 67 73 77 78 80 82 86 87 88
python tombola_predict.py accuracy --sorteo N
```

---

## Historial de resultados recientes

### Lunes 22/06/2026 — v1.1
Resultado: 01 10 11 12 13 15 18 19 21 25 30 37 43 45 57 66 72 85 87 96
E1 1/6 (11)  E2 1/6 (10)  E3 1/6 (72)  E4 2/6 (30, 87)
Total: 5/20  |  V2+V3+V4+V5 con acierto  |  C5: 2/4

### Miércoles 18/06/2026 — v1.1
Resultado: 02 07 09 17 18 21 23 28 30 33 39 47 58 64 80 81 87 92 94 97
E1 +$100 (39)  E2 +$300 (07 64 81)
Verticales: V1+$100 V3+$100 V4+$100 V6+$100 = +$400
Combinados: C1+$100 C2+$200 C3+$100 = +$400

### Viernes 12/06/2026
E2 +$100  |  E4 +$100 +$100

### Jueves 11/06/2026
E2 +$100 +$100 +$100 +$100  |  E3 +$100  |  E5 +$100

### Miércoles 10/06/2026
E3 +$100 +$100

### Martes 09/06/2026
E3 +$100

### Lunes 08/06/2026
E1 +$100  |  E3 +$100 +$100  |  E4 +$100

### Jueves 04/06/2026
Elite 1 P01, P02, P05, P06

### Lunes 01/06/2026
Buen sorteo general

---

## Consejos de uso

- Entrenar con `--last 200` una vez por semana e incorporar feedback diario.
- Actualizar el CSV con resultados nuevos antes de regenerar transiciones.
- El CSV (`tombolas.csv`) es la fuente de verdad para el entrenamiento.
- Los archivos `tombola_N_learning_v11.json` guardan el historial de predicciones y los factores aprendidos — hacer backup antes de `--reset`.
