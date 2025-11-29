from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    return _model.encode(text, convert_to_numpy=True)

def get_embeddings(texts: List[str]) -> np.ndarray:
    return _model.encode(texts, convert_to_numpy=True)
