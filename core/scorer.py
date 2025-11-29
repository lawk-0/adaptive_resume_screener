import numpy as np
from typing import Dict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_score(features: Dict) -> float:
    """
    features: dictionary with keys like "similarity", "skill_match", etc.
    For now, just scale similarity into 0–100.
    """
    sim = features.get("similarity", 0.0)
    # similarity is usually between 0 and 1
    score = sim * 100
    return round(score, 2)
