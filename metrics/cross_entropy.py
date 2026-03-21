import numpy as np

def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> float:
    loss = 0
    n = y.shape[0]
    epsilon = 1e-15
    for i in range(n):
        p_i = np.clip(y_pred[i], epsilon, 1 - epsilon)

        current_loss = (y[i] * np.log(p_i)) + ((1 - y[i]) * np.log(1 - p_i))

        loss += current_loss
    return - (loss / n)