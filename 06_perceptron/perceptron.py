import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            lr: float,
            max_iters: int,
            stats: bool) -> None:
        '''
        Train the Perceptron using Rosenblatt's learning rule.
        Updates weights and bias only when a misclassification occurs.
        Stops early if no errors are made in a full epoch.
        Only guaranteed to converge if data is linearly separable.

         Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)            
        y: np.ndarray of shape (n_samples, 1)
        lr: float learning rate that dictate how much gradients influence
            weights and bias
        max_iters: int amount of epochs
        stats: bool showing loss every iteration
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
             return None
        lr = float(lr)
        max_iters = int(max_iters)

        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:     
            y = np.reshape(y, (-1, 1))
                
        self.weights = np.random.rand(1, X.shape[1])
        self.bias = 0

        for _ in range(max_iters):
            errors = 0
            for Xi, yi in zip(X, y):
                output = np.sign(self.weights @ Xi.T + self.bias)

                if output != yi:
                    self.weights = self.weights + lr * yi * Xi
                    self.bias = self.bias + lr * yi
                    errors += 1

            if stats:
                print(f'Epoch: {_}, error: {errors}')

            if errors == 0:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict class labels for samples in X.
        Returns -1 or 1 for each sample based on the sign
        of the weighted sum of inputs.

         Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        '''
        res = []
        for Xi in X:
            output = np.sign(self.weights @ Xi.T + self.bias)
            res.append(output)
        return res