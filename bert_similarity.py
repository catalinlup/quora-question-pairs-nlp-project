from typing import List
import numpy as np


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def predict_similarity(embeddings1, embeddings2, threshold):
    similarity = cosine_similarity(embeddings1, embeddings2)
    return (similarity >= threshold).astype(np.int32)



