from abc import ABC, abstractmethod
import numpy as np

class Linear_Regression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        pass

