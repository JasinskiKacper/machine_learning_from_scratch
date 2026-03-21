import numpy as np
from abc import abstractmethod


class Logistic_Regression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z: np.ndarray) -> float:
        '''
        Compute the sigmoid activation function

        Parameters
        ----------
        z: result of z = Xw + b can be single number or an array
        '''
        return 1 / (1 + np.exp(-z))

    @abstractmethod
    def fit(self):
        pass

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        '''
        Predicting the most probability result of predicted sample
        and returning probability 0 = 0% 1 = 100%
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        '''
        X = np.asarray(X).astype(float)
        X = X.reshape((-1, 1))
        z = (X @ self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict_classes(self, X:np.ndarray) -> np.ndarray:
        '''
        Predicting the most probability result of predicted sample
        and returning result True/False
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        '''
        X = np.asarray(X).astype(float)
        X = X.reshape((-1, 1))
        z = (X @ self.weights) + self.bias
        return self.sigmoid(z) > 0.5
