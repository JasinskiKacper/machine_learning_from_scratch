import numpy as np
from linear_regression import Linear_Regression, RMSE

class Linear_Regression_Polynomial(Linear_Regression):

    def transform(self, 
                  X: np.ndarray, 
                  power: int) -> np.ndarray:
        '''
        transform expands X by its powers (X^1, X^2, ..., X^power).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        power : int of polynomial regression

        Returns:
            X_poly : np.ndarray of shape(n_samples, n_features * power)
                Matrix containing X raised to powers
        '''
        return np.column_stack([X**i for i in range(1, power + 1)])