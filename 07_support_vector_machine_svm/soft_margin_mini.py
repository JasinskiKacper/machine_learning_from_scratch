import numpy as np
from SVM import Support_Vector_Machine
from soft_margin_gd import hinge_loss_C

class Soft_margin_MINI(Support_Vector_Machine):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool,
            C: float) -> None:
        '''
        Soft margin SVM trained using Mini-Batch Gradient Descent.
        Splits data into batches of 16 samples, shuffled each epoch.
        Balances stability of batch GD with speed of SGD.
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

        batch_size = 16
        batches = np.arange(0, y.shape[0], batch_size)

        for _ in range(max_iters):
            np.random.shuffle(batches)
            for batch in batches:
                k = len(X[batch: batch + batch_size])
                X_batch = X[batch: batch + batch_size]
                y_batch = y[batch: batch + batch_size]
                
                s =  X_batch @ self.weights + self.bias
                m =  y_batch * s
                m = m.ravel()

                mask = m < 1
                if len(y_batch[mask]) > 0:
                    dw = self.weights - C * (X_batch[mask].T @ y_batch[mask]) / k
                    db = - C * np.sum(y_batch[mask]) / k
                else:
                    dw = self.weights
                    db = 0

                self.weights = self.weights - (lr * dw)
                self.bias = self.bias - (lr * db)

            if stats:
                print(f'For {_} epochs loss:\n{hinge_loss_C(X, y, self.weights, self.bias, C)}')

