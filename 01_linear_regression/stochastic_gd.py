import numpy as np
from linear_regression import Linear_Regression, RMSE

class Linear_Regression_Stochastic_GD(Linear_Regression):
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool = False) -> None:
        '''
        Fit linear regression using stochastic gradient descent (SGD)
        by shuffling data max_iters times and for each time iteration
        method process every sample, computes the gradient and updates
        the weights and bias.

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
        
        for _ in range(max_iters):
            # Creating list of indexes and shuffle it
            index = np.arange(n)
            np.random.shuffle(index)
            
            for i in index:
                Xi = X[i:i+1, :] # Xi.shape() = (1, n_features)
                yi = y[i:i+1, :] # yi.shape() = (1, 1)

                yi_pred = Xi @ self.weights + self.bias
                error = yi_pred - yi


                dw = 2 * Xi.T @ error
                db = 2 * np.sum(error)

                self.weights = self.weights - (lr * dw)
                self.bias = self.bias - (lr * db)
            
            if stats:
                print(f'RMSE: {RMSE(yi, yi_pred)}')