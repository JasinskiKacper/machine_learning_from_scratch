from linear_regression import Linear_Regression
import numpy as np

class Linear_Regression_Gradient(Linear_Regression):

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray) -> None:

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        
        if X.ndim == 1:
            X = np.reshape(X, (X.shape[0], 1))

        