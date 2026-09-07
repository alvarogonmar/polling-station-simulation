# Sistema de afinidad por plataforma política — resumen para el equipo

Este documento explica la función nueva que se agregó al modelo de simulación: un sistema de **afinidad entre las prioridades de cada votante y la plataforma de cada partido**, que se suma como un factor más a la decisión de voto que ya existía. Está organizado punto por punto, con el código y la dirección exacta de cada archivo, para que cualquiera del equipo pueda entenderlo y probarlo sin haber estado presente cuando se construyó.

---

## Punto 1 — Qué problema resuelve y qué NO hace

El modelo original decidía el partido de cada votante (`get_party_probabilities` en `polling_station_model.py`) combinando **cuatro factores**: grupo de edad, educación, ingreso e ideología. Cada uno suma o resta a un "puntaje de utilidad" por partido, y ese puntaje pasa por un softmax que da la probabilidad final de votar por cada uno de los 5 partidos (P1 a P5).

Lo que se agregó es un **quinto factor**: qué tanto coincide lo que le importa a cada votante (salud, economía, seguridad, etc.) con lo que ofrece cada partido en su plataforma. Este factor se **suma** a los cuatro que ya existían — no reemplaza nada, no cambia cómo funcionaban los otros cuatro, y se puede apagar por completo con un interruptor (ver Punto 5).

---

## Punto 2 — Los 7 temas de política pública y las plataformas de los partidos

**Archivo:** `config.py`

Se definieron 7 temas (`ISSUES`) y se le asignó a cada partido 100 puntos repartidos entre ellos, dándole a cada uno una identidad política reconocible:

```python
ISSUES = [
    "health", "economic_support", "family_support",
    "education", "security", "civil_rights", "environment",
]

# 100 puntos por partido, identidades políticas distinguibles:
# P1 económico/populista, P2 salud/familia, P3 centrista, P4 progresista, P5 seguridad
PARTY_PLATFORMS = {
    "P1": [15, 35, 20, 10,  5,  5, 10],
    "P2": [30, 15, 25, 15,  5,  5,  5],
    "P3": [15, 15, 15, 15, 15, 10, 15],
    "P4": [10,  5, 10, 20,  5, 25, 25],
    "P5": [10, 20,  5,  5, 40,  5, 15],
}
```

Cada lista sigue el mismo orden que `ISSUES`. Por ejemplo, P1 le da 35 de sus 100 puntos a `economic_support` (economía) — es el partido "económico/populista". P5 le da 40 a `security` — es el partido de seguridad.

---

## Punto 3 — El vector de prioridades de cada votante (`issue_weights`)

**Archivo:** `mesa_agents.py`, clase `VoterAgent`

Cada votante genera su propio vector de 7 números (que también suma 100), a partir de sus rasgos: ingreso, educación, edad, ideología, y si tiene hijos. Parte de una base pareja y le suma/resta efectos por categoría:

```python
def _compute_issue_weights(self):
    if not config.ENABLE_ISSUE_AFFINITY:
        return None

    weights = np.array(config.BASE_ISSUE_WEIGHTS, dtype=float)
    weights += np.array(config.ISSUE_INCOME_EFFECTS[self.income], dtype=float)
    weights += np.array(config.ISSUE_EDUCATION_EFFECTS[self.education], dtype=float)
    weights += np.array(config.ISSUE_AGE_GROUP_EFFECTS[self.age_group], dtype=float)
    weights += np.array(config.ISSUE_IDEOLOGY_EFFECTS, dtype=float) * self.ideology
    if self.has_children:
        weights += np.array(config.ISSUE_HAS_CHILDREN_BONUS, dtype=float)

    weights = np.clip(weights, 0.05, None)
    weights = weights / weights.sum() * 100.0

    rounded = [round(float(w), 2) for w in weights]
    rounded[-1] = round(rounded[-1] + round(100.0 - sum(rounded), 2), 2)
    return rounded
```

Ejemplo real: un votante de ingreso bajo, sin hijos, terminó con `economic_support = 19.11` como su prioridad más alta. Un votante de ingreso alto **con hijos** terminó con `family_support` como la más alta. Esto es exactamente lo que se buscaba: que el vector de cada votante refleje sus propias circunstancias.

### Punto 3.1 — El atributo nuevo `has_children`

También en `mesa_agents.py`, se agregó un atributo booleano nuevo, generado con una probabilidad (Bernoulli) que depende del grupo de edad:

```python
def _decide_has_children(self):
    if not config.ENABLE_ISSUE_AFFINITY:
        return False  # no consume rng: el resto de la simulación no se ve afectado

    probability = config.HAS_CHILDREN_PROBABILITY_BY_AGE_GROUP[self.age_group]
    return bool(self.model.rng.random() < probability)
```

```python
# config.py
HAS_CHILDREN_PROBABILITY_BY_AGE_GROUP = {
    "18-29": 0.20, "30-44": 0.65, "45-59": 0.45, "60+": 0.10,
}
```

Se agrega porque el tema "apoyos para niños/familia" (`family_support`) solo tiene sentido evaluarlo por votante si sabemos si tiene hijos o no.

---

## Punto 4 — Cómo se calcula la afinidad: distancia L1 invertida

**Archivo:** `polling_station_model.py`, método nuevo `get_issue_affinity_scores`

Con el vector del votante y la plataforma de cada partido (ambos suman 100, misma escala), se compara qué tan parecidos son restando punto por punto y sumando los valores absolutos — la **distancia L1** o "distancia Manhattan". El rango posible es de 0 (idénticos) a 200 (completamente opuestos). Se invierte a una similitud entre 0 y 1:

```python
def get_issue_affinity_scores(self, voter):
    if not config.ENABLE_ISSUE_AFFINITY or voter.issue_weights is None:
        return np.zeros(len(self.parties), dtype=float)

    voter_vector = np.array(voter.issue_weights, dtype=float)
    affinities = []
    for party in self.parties:
        platform = np.array(config.PARTY_PLATFORMS[party], dtype=float)
        l1_distance = np.abs(voter_vector - platform).sum()
        affinities.append(1.0 - (l1_distance / 200.0))
    return np.array(affinities, dtype=float)
```

**¿Por qué L1 y no similitud coseno (cosine similarity)?** Cosine similarity resuelve diferencias de *magnitud* entre vectores — pero aquí ambos vectores ya suman 100, misma escala, ese problema no existe. Además, con vectores de valores no-negativos, cosine similarity queda comprimida en un rango angosto (~0.85–0.99), lo que aplastaría diferencias reales entre plataformas. L1 da una lectura directa y explicable: "cuántos de los 100 puntos del votante están mal alineados con el partido", en un rango exacto y fácil de interpretar.

---

## Punto 5 — Cómo se combina con la decisión que ya existía (y el interruptor)

**Archivo:** `polling_station_model.py`, método `get_party_probabilities` (ya existente, modificado)

```python
def get_party_probabilities(self, voter):
    age_base = np.array(config.AGE_GROUP_PROBABILITIES[voter.age_group], dtype=float)
    education_effect = np.array(config.EDUCATION_EFFECTS[voter.education], dtype=float)
    income_effect = np.array(config.INCOME_EFFECTS[voter.income], dtype=float)
    ideology_effect = np.array(config.IDEOLOGY_EFFECTS, dtype=float) * voter.ideology

    # --- issue-based affinity system (bloque activable/desactivable) ---
    affinity_scores = self.get_issue_affinity_scores(voter)
    affinity_effect = (affinity_scores - 0.5) * config.AFFINITY_WEIGHT if config.ENABLE_ISSUE_AFFINITY else affinity_scores
    # --------------------------------------------------------------------

    utility_scores = (
        age_base + education_effect + income_effect + ideology_effect + affinity_effect
    )
    exp_scores = np.exp(utility_scores - np.max(utility_scores))
    probabilities = exp_scores / exp_scores.sum()
    return probabilities
```

Los 4 términos originales siguen exactamente igual. Se agregó un quinto término (`affinity_effect`) que se **suma** antes del softmax — igual que los otros, no en su lugar.

### El interruptor: `ENABLE_ISSUE_AFFINITY`

**Archivo:** `config.py`, primera constante del bloque nuevo:

```python
# =====================================================================
# ISSUE-BASED AFFINITY SYSTEM (activable/desactivable como bloque)
# Para desactivar por completo esta función y volver al comportamiento
# original (byte por byte), poner ENABLE_ISSUE_AFFINITY = False.
# =====================================================================
ENABLE_ISSUE_AFFINITY = True
```

Con `False`:
- `_decide_has_children()` regresa `False` de inmediato, **sin tocar el generador de números aleatorios** (`self.model.rng`) — esto es clave, porque si se consumiera una llamada de más al azar, se correrían todos los números aleatorios siguientes de la simulación (paciencia, tiempos de servicio, etc.).
- `_compute_issue_weights()` regresa `None`.
- `affinity_effect` se vuelve un vector de puros ceros, así que sumarlo no cambia nada.

**Se comprobó, comparando contra el código original línea por línea (`git stash`), que con el interruptor apagado la simulación es idéntica byte por byte a como era antes de este cambio** — la única diferencia son dos claves nuevas e inertes en el JSON (`has_children: false`, `issue_weights: null`). Apagar el interruptor es un "volver exactamente a como estaba" real, no una aproximación.

---

## Punto 6 — El hallazgo durante las pruebas: un partido se comía a los demás

Al probar la función con los primeros números que se habían propuesto, se encontró un problema de **calibración** (no un bug de código): entre más peso se le daba al nuevo término, más votos se concentraban en un solo partido (P3) en vez de repartirse de forma realista.

**Causa:** la plataforma de P3 (`[15,15,15,15,15,10,15]`, muy "plana") resultó casi idéntica al vector *promedio* de cualquier votante, porque los efectos por atributo (ingreso, educación, edad) eran muy chicos comparados con la base uniforme. Casi todos los votantes generaban vectores muy parecidos entre sí, y ese promedio se parecía mucho a P3.

| | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| Sin afinidad (referencia) | 19.2% | 24.6% | 39.2% | 15.6% | 1.4% |
| Con afinidad — versión inicial | 17.8% | 13.0% | **61.2%** ⚠️ | 8.0% | **0.0%** ⚠️ |
| Con afinidad — versión final | 22.0% | 14.6% | 46.2% | 16.6% | 0.6% |

**Solución:** se multiplicaron por 6 las magnitudes de los efectos por atributo (`ISSUE_INCOME_EFFECTS`, `ISSUE_EDUCATION_EFFECTS`, `ISSUE_AGE_GROUP_EFFECTS`, `ISSUE_IDEOLOGY_EFFECTS`, `ISSUE_HAS_CHILDREN_BONUS`, todos en `config.py`), para que los votantes se separen más del "centro" y puedan acercarse de verdad a plataformas más extremas. Con esto, ningún partido queda en 0%, y el efecto del nuevo término es real (~21% de los votantes cambian de partido) sin que ninguno domine de forma irreal.

---

## Punto 7 — Verificación completa que se hizo

| Prueba | Resultado |
|---|---|
| Cada plataforma de partido suma exactamente 100 | ✅ |
| `issue_weights` de cada votante suma 100, sin valores negativos | ✅ |
| Coherencia individual (bajo ingreso → prioriza economía; con hijos → prioriza familia) | ✅ confirmado aislando cada variable |
| Determinismo: dos corridas con el interruptor prendido dan resultados idénticos | ✅ (gracias a `SEED = 42` fijo) |
| Interruptor apagado = comportamiento original, comparado línea por línea contra el código previo al cambio | ✅ idéntico byte por byte |
| Smoke test de la API real (`/`, `/get_agents`, `/get_results`, `/reset`) | ✅ 200, sin excepciones, campos nuevos presentes y campos viejos intactos |

---

## Punto 8 — Cómo probarlo ustedes mismos

```bash
# 1. Ver que las plataformas suman 100
python -c "import config; [print(p, sum(v)) for p, v in config.PARTY_PLATFORMS.items()]"

# 2. Correr el modelo y ver los campos nuevos en la consola
python polling_station_model.py

# 3. Levantar la API y ver un votante completo
python polling_station_api.py
curl http://127.0.0.1:5000/get_agents
```

Para desactivar la función por completo (por ejemplo, para comparar contra el comportamiento de antes), solo hay que cambiar una línea en `config.py`:

```python
ENABLE_ISSUE_AFFINITY = False
```

---

## Punto 9 — Archivos que se modificaron

- **`config.py`** — bloque nuevo al final: interruptor, temas, plataformas de partidos, y todos los efectos por atributo.
- **`mesa_agents.py`** — atributos nuevos `has_children` e `issue_weights` en `VoterAgent`, dos métodos nuevos, y dos claves nuevas en `to_dict()`.
- **`polling_station_model.py`** — método nuevo `get_issue_affinity_scores`, y el término de afinidad sumado en `get_party_probabilities`.
- **`requirements.txt`** — se fijó `mesa<3` (sin esto, el proyecto no arranca con versiones nuevas de Mesa, que eliminaron la API que usa este código).

**Fuera de alcance por ahora** (no se tocó): el cliente de Unity (`SimulationApiClient.cs`) sigue igual — como usa `JsonUtility`, ignora en silencio las claves nuevas del JSON, así que Unity sigue funcionando exactamente igual sin cambios. Si en algún momento quieren visualizar `has_children`/`issue_weights` del lado de Unity, habría que agregar esos dos campos a la clase `AgentData` en ese archivo.
