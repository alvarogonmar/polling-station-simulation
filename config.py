"""Configuration values for the polling station simulation."""

SEED = 42

N_VOTERS = 500
PARTIES = ["P1", "P2", "P3", "P4", "P5"]
N_POLL_WORKERS = 3
N_BALLOT_BOXES = 5

P_TURNOUT = 0.65
P_NULL_VOTE = 0.03
P_ID_INVALID = 0.01

SIMULATION_STEPS = 360
QUEUE_THRESHOLD = 20
POWER_FAILURE_STEP = 120
POWER_FAILURE_DURATION = 15

ARRIVAL_START_STEP = 1
ARRIVAL_END_STEP = 300
MEAN_INTERARRIVAL_TIME = 0.9
MEAN_SERVICE_TIME = 1.2
MIN_VOTING_TIME = 5
MAX_VOTING_TIME = 9
MIN_PATIENCE = 15
MAX_PATIENCE = 45 # paciencia 

AGE_GROUP_PROBABILITIES = {
    "18-29": [0.35, 0.25, 0.20, 0.10, 0.10],
    "30-44": [0.25, 0.30, 0.20, 0.15, 0.10],
    "45-59": [0.20, 0.25, 0.30, 0.15, 0.10],
    "60+": [0.15, 0.20, 0.35, 0.20, 0.10],
}

EDUCATION_LEVELS = ["low", "medium", "high"]
INCOME_LEVELS = ["low", "medium", "high"]

EDUCATION_EFFECTS = {
    "low": [0.05, 0.00, 0.05, -0.05, -0.05],
    "medium": [0.00, 0.05, 0.00, 0.05, -0.05],
    "high": [-0.05, 0.05, -0.05, 0.10, 0.05],
}

INCOME_EFFECTS = {
    "low": [0.10, 0.05, 0.00, -0.05, -0.10],
    "medium": [0.00, 0.05, 0.05, 0.00, -0.05],
    "high": [-0.10, -0.05, 0.05, 0.10, 0.10],
}

IDEOLOGY_EFFECTS = [-0.20, -0.10, 0.00, 0.10, 0.20]

# =====================================================================
# ISSUE-BASED AFFINITY SYSTEM (activable/desactivable como bloque)
# Para desactivar por completo esta función y volver al comportamiento
# original (byte por byte), poner ENABLE_ISSUE_AFFINITY = False.
# =====================================================================
ENABLE_ISSUE_AFFINITY = True

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

BASE_ISSUE_WEIGHTS = [1.0] * 7

# NOTA: estos efectos se probaron empíricamente y se multiplicaron x6 respecto
# a la primera versión — con la magnitud original, los vectores de los votantes
# quedaban demasiado parecidos al promedio, y ese promedio coincide casi
# exactamente con la plataforma "plana" de P3, causando que P3 acaparara el
# voto (61%+) en vez de repartirse de forma realista entre los 5 partidos.
ISSUE_INCOME_EFFECTS = {
    "low":    [ 0.00,  2.10,  0.30,  0.00, -0.30, -0.30, -0.60],
    "medium": [ 0.00,  0.60,  0.00,  0.00,  0.00,  0.00,  0.00],
    "high":   [ 0.00, -1.20, -0.30,  0.30,  0.30,  0.30,  0.90],
}
ISSUE_EDUCATION_EFFECTS = {
    "low":    [ 0.30,  0.30,  0.30, -0.60,  0.30, -0.60, -0.60],
    "medium": [ 0.00,  0.00,  0.00,  0.30,  0.00,  0.00,  0.00],
    "high":   [-0.30, -0.30, -0.30,  0.90, -0.30,  0.90,  0.90],
}
ISSUE_AGE_GROUP_EFFECTS = {
    "18-29": [-0.30,  0.00, -0.60,  0.60, -0.60,  0.60,  0.60],
    "30-44": [ 0.00,  0.30,  0.90,  0.00,  0.00,  0.00, -0.30],
    "45-59": [ 0.30,  0.30,  0.00, -0.30,  0.30, -0.30, -0.30],
    "60+":   [ 1.20,  0.00, -0.60, -0.60,  0.90, -0.60, -0.60],
}
ISSUE_IDEOLOGY_EFFECTS = [-0.30, -0.60, -0.30, 0.00, 0.90, -0.60, -0.60]
ISSUE_HAS_CHILDREN_BONUS = [0.00, 0.00, 1.80, 0.30, 0.00, 0.00, 0.00]

HAS_CHILDREN_PROBABILITY_BY_AGE_GROUP = {
    "18-29": 0.20, "30-44": 0.65, "45-59": 0.45, "60+": 0.10,
}

AFFINITY_WEIGHT = 0.4  # peso del término de afinidad dentro del softmax
