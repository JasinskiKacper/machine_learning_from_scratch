import numpy as np
from logistic_regression import Logistic_Regression

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import cross_entropy

class Logistic_Regression_MiniBatch(Logistic_Regression):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool = False) -> None:
        '''

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
            X = X.reshape([-1, 1]) # (n_samples, n_features)
        if y.ndim == 1:
            y = y.reshape([-1, 1]) # (n_samples, 1)

        self.weights = np.zeros([X.shape[1], 1]) # (n_features, 1)
        self.bias = 0

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        batch_size = 32
        batches = np.arange(0, X.shape[0], batch_size, dtype=int)
        
        for _ in range(max_iters):
            np.random.shuffle(batches)
            for batch in batches:
                n = len(X[batch: (batch + batch_size)])
                X_batch = X[batch: (batch + batch_size)]
                y_batch = y[batch: (batch + batch_size)]
        
                z = X_batch @ self.weights + self.bias
                y_pred = self.sigmoid(z)

                dw = X_batch.T @ (y_pred - y_batch) / n
                db = np.sum(y_pred - y_batch) / n

                self.weights = self.weights - (lr * dw)
                self.bias = self.bias - (lr * db)

            if stats:
                z = X @ self.weights + self.bias
                y_pred = self.sigmoid(z)
                loss = cross_entropy(y, y_pred)
                print(f'For {_} epoch\nLoss: {loss}')