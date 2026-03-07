import numpy as np
from linear_regression import Linear_Regression, RMSE


class Linear_Regression_Gradient(Linear_Regression):

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            max_iter: int,
            lr: float,
            stats: bool = False) -> None:
        '''
        Fit linear regression using the gradient descent method by
        calculating predictions, error, and derivatives of weights and bias,
        starting from random values and subtracting the error multiplied
        by the learning rate in every iteration.

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
        w = np.random.rand(X.shape[1], 1)
        b = np.random.uniform(-0.01, 0.01)
        n = y.size

        for _ in range(max_iter):
            y_pred = X @ w + b

            error = y_pred - y

            if stats:
                print(f'RMSE: {RMSE(y, y_pred)}')
            
            # Compute gradients of the loss function
            dw = (2 / n) * X.T @ (error)
            db = (2 / n) * np.sum(error)

            # Updating weights and bias
            w = w - (lr * dw)
            b = b - (lr * db)
        
        self.weights = w
        self.bias = b