import numpy as np

def MSE(y_true: np.ndarray, y_approx: np.ndarray) -> float:
    if not isinstance(y_true, np.ndarray) or not isinstance(y_approx, np.ndarray):
        raise TypeError
    if y_true.shape[0] != y_approx.shape[0]:
        raise ValueError
    if y_true.size == 0 or y_approx.size == 0:
        raise ValueError
    
    n = y_true.shape[0]
    error_sum = 0

    for i in range(n):
        error_sum += (y_true[i] - y_approx[i]) ** 2

    return error_sum / n