import numpy as np


def embedding_similarity(embedding1, embedding2):
    score = np.dot(
        embedding1,
        embedding2
    ) / (
        np.linalg.norm(embedding1)
        * np.linalg.norm(embedding2)
    )
    return float(score)
