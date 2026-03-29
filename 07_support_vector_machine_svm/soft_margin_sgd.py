import numpy as np
from SVM import Support_Vector_Machine
from soft_margin_gd import hinge_loss_C

class Soft_margin_SGD(Support_Vector_Machine):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool,
            C: float) -> None:
        '''
        Soft margin SVM trained using Stochastic Gradient Descent.
        Processes one random sample per update, faster convergence than batch GD
        but noisier updates. Shuffles data each epoch to avoid cyclical patterns.
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples, 1)
        max_iters: int numbers of maximum iterations 
        lr: float learning rate that dictate how much gradients influence
            weights and bias
        stats: bool showing loss every iteration
        C: float penalty parameter - high C = narrow margin, low C = wide margin
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

        shuf = np.arange(0, y.shape[0], 1, dtype=int)
        for _ in range(max_iters):
            np.random.shuffle(shuf)
            for i in shuf:
                si = X[i] @ self.weights + self.bias
                mi = y[i] * si

                if mi >= 1:
                    dw = self.weights
                    db = 0
                else:
                    dw = self.weights - C * y[i] * X[i].reshape(-1, 1)
                    db = - C * y[i]

                self.weights = self.weights - (lr * dw)
                self.bias = self.bias - (lr * db)
            
            if stats:
                print(f'For {_} epochs loss:\n{hinge_loss_C(X, y, self.weights, self.bias, C)}')

