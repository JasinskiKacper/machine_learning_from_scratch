import numpy as np
from linear_regression import Linear_Regression, RMSE

class Linear_Regression_SVD(Linear_Regression):
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray) -> None:
        '''
        Fit linear regression using the Singular Value Decomposition (SVD).
        This method computes the least squares solution using the
        pseudoinverse obtained from the SVD decomposition
            X = U Σ V^T
        The regression weights are computed as
            w = V Σ^{-1} U^T y

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples, )
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError
        
        if X.ndim == 1: # changing shape from (n, ) to (n, 1)
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        
        # adding column of ones for bias
        X_bias = np.c_[np.ones(X.shape[0]), X]

        # decompositing X
        U, E, Vt = np.linalg.svd(X_bias, full_matrices=False)

        X = Vt.T @ np.linalg.inv(np.diag(E)) @ U.T @ y
        
        self.weights = X[1:]
        self.bias = X[0]