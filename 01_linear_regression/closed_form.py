import numpy as np
from linear_regression import Linear_Regression

class Linear_Regression_Closed_Form(Linear_Regression):

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray) -> None:
        
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        
        if X.ndim == 1: # changing shape from (n, ) to (n, 1)
            X = np.reshape(X, (X.shape[0], 1))
        
        X = np.c_[np.ones(X.shape[0]), X]
        
        # w = (X^T @ X)^-1 @ X^T @ y 
        # solving Ax = b for stability
        A = X.T @ X
        b = X.T @ y
        x = np.linalg.solve(A, b)
        
        self.bias = x[0]
        self.weights = x[1:]