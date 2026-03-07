import numpy as np
from .mse import MSE

def RMSE(y_true: np.ndarray, y_approx: np.ndarray) -> float:
    return np.sqrt(MSE(y_true, y_approx))