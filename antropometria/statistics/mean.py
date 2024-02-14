import numpy as np


def calculate_mean_from_dict(scores_dict: dict) -> dict:
    mean_score = {}
    for score_key, score_value in scores_dict.items():
        mean_score[score_key] = np.mean(score_value, axis=0)

    return mean_score
