import numpy as np

def compute_retrieval_confidence(results):
    """
    Measures how strong top-k matches are.
    Higher score = better retrieval confidence.
    """

    distances = results.get("distances", [[]])[0]

    if not distances:
        return 0.0

    # Convert distance → similarity (approx)
    similarities = [1 / (1 + d) for d in distances]

    return float(np.mean(similarities))


def compute_relevance_spread(results):
    distances = results.get("distances", [[]])[0]

    if len(distances) < 2:
        return 0.0

    return float(np.std(distances))



def retrieval_quality_score(results):
    confidence = compute_retrieval_confidence(results)
    spread = compute_relevance_spread(results)

    score = confidence - (spread * 0.1)

    return round(score, 3)