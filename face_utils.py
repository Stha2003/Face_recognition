import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_best_match(embedding, known_embeddings, known_names, threshold=0.6):
    similarities = [cosine_similarity(embedding, e) for e in known_embeddings]
    best_match = np.argmax(similarities)
    if similarities[best_match] >= threshold:
        return known_names[best_match]
    return "Unknown"