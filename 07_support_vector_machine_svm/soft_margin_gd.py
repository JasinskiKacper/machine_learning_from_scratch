import numpy as np
from SVM import Support_Vector_Machine

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import hinge_loss_C

class Soft_margin_GD(Support_Vector_Machine):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool,
            C: float) -> None:
        '''
        Soft margin SVM trained using Gradient Descent.
        Unlike hard margin SVM, allows some samples to violate the margin,
        controlled by the penalty parameter C.
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples, 1)
        max_iters: int numbers of maximum iterations 
        lr: float learning rate that dictate how much gradients influence
            weights and bias
        stats: bool showing loss every iteration
        c: float penalty parameter controlling trade-off between margin width
           and classification errors. High C = narrow margin, few errors (risk of overfitting).
           Low C = wide margin, more errors tolerated (better generalization).
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        max_iters = int(max_iters)
        lr = float(lr)
        C = float(C)

        if X.ndim == 1:
            X = X.reshape(-1, 1) # (n_samples, n_features) 
        if y.ndim == 1:
            y = y.reshape(-1, 1) # (n_samples, 1)

        self.weights = np.random.randn(X.shape[1], 1) * 0.01 # (n_features, 1)
        self.bias = 0

        for _ in range(max_iters):
            s = X @ self.weights + self.bias

            m = y * s
            index = []
            for i in range(m.shape[0]):
                if m[i] < 1:
                    index.append(i)
            
            dw = self.weights - C * (X[index].T @ y[index]) / X.shape[0]
            db = - C * np.sum(y[index])

            self.weights = self.weights - lr * dw
            self.bias = self.bias - lr * db

            if stats:
                print(f'For {_} epochs loss:\n{hinge_loss_C(X, y, self.weights, self.bias, C)}')

