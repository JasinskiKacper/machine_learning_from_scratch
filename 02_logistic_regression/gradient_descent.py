import numpy as np
from logistic_regression import Logistic_Regression

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import cross_entropy


class Logistic_Regression_GradientDescent(Logistic_Regression):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool = False) -> None:
        '''
        Logistic Regression model trained using the Gradient Descent optimization algorithm.
        Inherits basic functionality (like sigmoid) from the Logistic_Regression base class.
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples, 1)
        max_iters: int numbers of maximum iterations 
        lr: flaot learning rate that dictate how much gradients influence
            weights and bias
        stats: bool showing loss every iteration
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        lr = float(lr)
        max_iters = int(max_iters)
        
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.weights = np.zeros((X.shape[1], 1)) # (n_features, 1)
        self.bias = 0 
        n = X.shape[0]
        for _ in range(max_iters):
            z = X @ self.weights + self.bias # (n_samples, 1)
            y_pred = self.sigmoid(z) # (n_samples, 1) n_samples[i] == [0, 1]

            loss = cross_entropy(y, y_pred)
            if stats:
                print(f'for {_} epochs\nLoss: {loss}')
            if loss < 1e-6:
                break
            dw = (X.T @ (y_pred - y)) / n # (n_features, 1)
            db = np.sum(y_pred - y) / n

            self.weights = self.weights - (lr * dw)
            self.bias = self.bias - (lr * db)