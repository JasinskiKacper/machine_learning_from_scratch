import numpy as np
from linear_regression import Linear_Regression, RMSE

class Linear_Regression_Mini_Batch(Linear_Regression):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats:bool = False) -> None:
        '''
        Fit linear regression using 

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples, )
        max_iter : int number of iterations
        lr : int controlling the step size of weights and bias updates
        stats : bool if True prints RMSE every iteration
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError
        
        # Changing shape from (n, ) to (n, 1)
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Initialize weights and bias with small random values
        self.weights = np.random.rand(X.shape[1], 1)
        self.bias = np.random.uniform(-0.01, 0.01)
        
        n = y.size
        batch_size = 32
        index = np.arange(n)

        for _ in range(max_iters):
            np.random.shuffle(index)

            for i in range(0, n-batch_size, batch_size):
                batch_indices = index[i : i + batch_size]
                # X_batch.size = (1, n_features), y_batch.size = (1, 1)
                X_batch = X[batch_indices]
                y_batch = y[batch_indices] 

                y_batch_pred = X_batch @ self.weights + self.bias

                error = y_batch_pred - y_batch

                dw = (2 / y_batch.shape[0]) * X_batch.T @ error
                db = (2 / y_batch.shape[0]) * np.sum(error)

                self.weights = self.weights - (lr * dw)
                self.bias = self.bias - (lr * db)
            
            if stats:
                print(f'RMSE: {RMSE(y_batch, y_batch_pred)}')