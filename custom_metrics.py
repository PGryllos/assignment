import numpy as np


def preciness(y_true, y_pred, threshold=.2):
    """Measures the percentage of times our predictions was in the
    range [true +/- true + (true * threshold)]
    """
    y_true_upper = y_true + y_true * threshold
    y_true_lower = y_true - y_true * threshold
    
    correct = np.logical_and(y_pred >= y_true_lower, y_pred <= y_true_upper).astype(int)
    return np.sum(correct) / correct.shape[0]