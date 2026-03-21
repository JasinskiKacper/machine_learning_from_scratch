import numpy as np
from logistic_regression import Logistic_Regression

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import cross_entropy

class Logistic_Regression_SG(Logistic_Regression):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool = False) -> None:
        '''
        This method optimizes weights and bias by updating them after processing 
        each individual training sample. Data is shuffled at the beginning of 
        each epoch to ensure stochasticity and better convergence.
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
        max_iters = int(max_iters)
        lr = float(lr)

        if X.ndim == 1:
            X = X.reshape([-1, 1]) # X = (n_samples, n_features)
        if y.ndim == 1: 
            y = y.reshape([-1, 1]) # y = (n_samples, 1)


        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        self.weights = np.zeros([X.shape[1], 1]) # w = (n_features, 1)
        self.bias = 0

        n = X.shape[0] 
        indices = np.arange(0, n, 1, dtype=int)

        for _ in range(max_iters):
            np.random.shuffle(indices)
            for i in indices:
                z = X[i] @ self.weights + self.bias
                y_pred = self.sigmoid(z)

                dw = (y_pred - y[i]) * X[i].reshape([-1, 1])
                db = (y_pred - y[i])

                self.weights = self.weights - (lr * dw)
                self.bias = self.bias - (lr * db)

            if stats:
                z_stats = X @ self.weights + self.bias
                y_stats = self.sigmoid(z_stats) 
                print(f'For {_} epoch\nLoss: {cross_entropy(y, y_stats)}')
