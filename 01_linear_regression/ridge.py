import numpy as np
from linear_regression import Linear_Regression, RMSE

class Linear_Regression_Ridge(Linear_Regression):

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            alpha: float) -> None:
        
        '''
        Fit linear regression using Ridge regression (L2 regularization).
        This method augments X with a column of ones for the bias term
        and adds a regularization term (punishment matrix) to penalize
        large weights, reducing overfitting. The bias term is not penalized.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples, )
        alpha : float higher values increase the penalty on large weights
            if alpha = 0,  ridge regression reduces to standard closed-form 
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError
        
        if X.ndim == 1: # changing shape from (n, ) to (n, 1)
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        X_bias = np.c_[np.ones(X.shape[0]), X]

        if alpha == 0:
            punish_matrix = 0
        else:
            punish_matrix = alpha * np.eye(X_bias.shape[1])
            punish_matrix[0, 0] = 0

        X = np.linalg.inv(X_bias.T @ X_bias + punish_matrix) @ X_bias.T @ y

        self.weights = X[1:]
        self.bias = X[0]