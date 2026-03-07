import numpy as np
from linear_regression import Linear_Regression

class Linear_Regression_Closed_Form(Linear_Regression):

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray) -> None:
        '''
        Fit linear regression using the closed-form solution.
        This method computes the optimal weights and bias using the formula:
            (X^T @ X)w = X^T @ y
        and solves for the linear model:
            y_pred = X @ w + b

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples, )
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError
        
        if X.ndim == 1: # changing shape from (n, ) to (n, 1)
            X = np.reshape(X, (X.shape[0], 1))
        
        # Add bias (column of ones)
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Closed form solution using above formula
        A = X_bias.T @ X_bias
        b = X_bias.T @ y
        result = np.linalg.solve(A, b)
        
        # Extract bias and weights
        self.bias = result[0]
        self.weights = result[1:]