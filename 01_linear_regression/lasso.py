import numpy as np
from linear_regression import Linear_Regression, RMSE


class Linear_Regression_Lasso(Linear_Regression):

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            alpha: float,
            max_iters: int,
            lr: float,
            stats: bool = False) -> None:
        
        '''
        Fit linear regression using Lasso regression (L1 regularization).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples, )
        alpha : float higher values increase the penalty on large weights
            if alpha = 0,  lasso reduces to cordinate descent
        max_iter : int number of iterations
        lr : int controlling the step size of weights and bias updates
        stats : bool if True prints RMSE every iteration
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError
        
        if X.ndim == 1: # changing shape from (n, ) to (n, 1)
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.weights = np.random.rand(X.shape[1], 1)
        self.bias = np.random.randint(-1, 1)
        n = X.shape[0]

        for _ in range(max_iters):
            for i in range((self.weights).shape[0]):
                y_pred = X @ self.weights + self.bias

                error = y - y_pred
                
                dwi = - (1 / n) * np.sum(X[:, i].T @ error)
                
                (self.weights)[i] = np.sign((self.weights)[i] - lr * dwi) * np.max((abs((self.weights)[i] - lr * dwi)) - lr * alpha, 0)
                
            db = (1 / n) * np.sum(error)
            self.bias = self.bias - (lr * db)
            if stats:
                print(f'RMSE: {RMSE(y, y_pred)}')

