from linear_regression import Linear_Regression
import numpy as np

class Linear_Regression_Gradient(Linear_Regression):

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            max_iter: int,
            lr: float) -> None:
        '''
        Args:
            X (np.ndarray): Data 
            y (np.ndarray): Data Labels
            max_iter (int):
            lr (int): 
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        
        if X.ndim == 1:
            X = np.reshape(X, (X.shape[0], 1))

        w = np.random.rand(X.shape[0], X.shape[1])
        b = np.random.uniform(-0.01, 0.01)

        for iter in range(max_iter):
            y_approx = X @ w + b
            error = y_approx - y
            