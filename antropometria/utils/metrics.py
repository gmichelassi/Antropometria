import math
import numpy as np


def calculate_mean(scores_dict: dict):
    mean_score = {}
    for score_key, score_value in scores_dict.items():
        mean_score[score_key] = np.mean(score_value, axis=0)
    return mean_score


def calculate_std(scores):
    mean = np.mean(scores)
    variance = 0
    for i in scores:
        variance += (i - mean) ** 2
    variance = variance / (len(scores) - 1)
    return math.sqrt(variance)
