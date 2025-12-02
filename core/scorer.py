import numpy as np
import joblib
from typing import Dict

# Load ML model
try:
    ML_MODEL = joblib.load("models/fit_model.pkl")
except Exception:
    ML_MODEL = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_score(features: Dict) -> float:
    """
    If ML model exists → use it.
    Otherwise → fallback to rule-based scoring.
    """
    sim = features.get("similarity", 0.0)
    skill_count = features.get("skill_count", 0)
    exp_years = features.get("experience_years", 0.0)

    if ML_MODEL:
        X = np.array([[sim, skill_count, exp_years]])
        proba = ML_MODEL.predict_proba(X)[0][1]  # probability of class 1 (good fit)
        return round(proba * 100, 2)

    # Fallback logic if model missing
    skill_factor = min(skill_count / 10.0, 1.0)
    exp_factor = min(exp_years / 5.0, 1.0)
    score = (0.60 * sim + 0.25 * skill_factor + 0.15 * exp_factor) * 100
    return round(score, 2)
