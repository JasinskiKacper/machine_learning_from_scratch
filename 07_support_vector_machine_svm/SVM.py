import numpy as np
from abc import abstractmethod

class Support_Vector_Machine:
    def __init__(self):
        self.weights = None
        self.bias = None

    @abstractmethod
    def fit(self):
        pass

    # default for linear models
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))

