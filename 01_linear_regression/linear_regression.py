import numpy as np

from abc import ABC, abstractmethod
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import MSE, RMSE

class Linear_Regression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1: # changing shape from (n, ) to (n, 1)
            X = np.reshape(X, (X.shape[0], 1))
        
        return X @ self.weights + self.bias

