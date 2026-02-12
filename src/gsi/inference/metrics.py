import numpy as np

def acc_fn(y_preds, y_true, k):
    top_k_preds = np.argsort(y_preds, axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return np.mean(correct)