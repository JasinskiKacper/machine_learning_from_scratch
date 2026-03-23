import numpy as np
from SVM import Support_Vector_Machine

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import hinge_loss 

class Hard_margin_GD(Support_Vector_Machine):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool) -> None:
        '''
        

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples, 1)
        max_iters: int numbers of maximum iterations 
        lr: float learning rate that dictate how much gradients influence
            weights and bias
        stats: bool showing loss every iteration
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        max_iters = int(max_iters)
        lr = float(lr)

        if X.ndim == 1:
            X = X.reshape(-1, 1) # (n_samples, n_features) 
        if y.ndim == 1:
            y = y.reshape(-1, 1) # (n_samples, 1)

        self.weights = np.random.randn(X.shape[1], 1) * 0.01 # (n_features, 1)
        self.bias = 0

        for _ in range(max_iters):
            for i in range(y.shape[0]):
                if y[i] * (X[i] @ self.weights + self.bias) >= 1:
                    dw = self.weights
                    db = 0
                else:
                    dw = self.weights - (y[i] * X[i].reshape(-1, 1))
                    db = - y[i]

                self.weights = self.weights - (lr * dw)
                self.bias = self.bias - (lr * db)
            if stats:
                loss = hinge_loss(X, y, self.weights, self.bias)
                print(f'For {_} epoch\nLoss: {loss}')