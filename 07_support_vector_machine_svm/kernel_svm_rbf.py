import numpy as np
from SVM import Support_Vector_Machine

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import dual_function

class Kernel_svm_RBF(Support_Vector_Machine):
    
    def RBF(self,
            X: np.ndarray,
            gamma: float
            ) -> np.ndarray:
        '''
        Computes the RBF (Gaussian) kernel matrix for the dataset.
        
        The RBF kernel between two points x and y is defined as:
        K(x, y) = exp(-gamma * ||x - y||^2)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data matrix.
        gamma : float
            The kernel coefficient. Higher values lead to more complex boundaries.
        '''
        n = X.shape[0]
        kernel_matrix = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                
                diff = X[i] - X[j]
                diff_sq = np.sum(diff ** 2)

                kernel_matrix[i, j] = np.exp(- gamma * diff_sq)
        
        return kernel_matrix
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            stats: bool,
            C: float,
            gamma: float) -> None:
        '''
        Trains the SVM model using the SMO algorithm to find optimal Lagrange multipliers (alphas).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples, 1)
            Target labels. Must be encoded as -1 and 1.
        max_iters : int
            Maximum number of passes through the dataset.
        stats : bool
            If True, prints the dual objective function value after each epoch.
        C : float
            Regularization parameter (Penalty). 
            Low C: Allows more margin violations (wider margin, prevents overfitting).
            High C: Strictly tries to classify all points correctly (narrower margin).
        gamma : float
            RBF kernel parameter. Defines the 'reach' of a single training example.
        '''
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        max_iters = int(max_iters)
        C = float(C)
        gamma = float(gamma)

        if X.ndim == 1:
            X = X.reshape(-1, 1) # (n_samples, n_features) 
        if y.ndim == 1:
            y = y.reshape(-1, 1) # (n_samples, 1)

        n = X.shape[0]
        self.bias = 0
        self.alpha = np.zeros(shape=(n, 1))
        # (n_samples, 1)

        kernel = self.RBF(X=X, gamma=gamma)
        # (n_samples, n_samples)
        index = np.arange(0, n, 1, dtype=int)

        for _ in range(max_iters):
            ay = (self.alpha * y).flatten()
            predictions = (kernel @ ay) + self.bias
            E = predictions - y.flatten()

            np.random.shuffle(index)
            for i in range(n):
                j = index[i]

                if j != i:
                    fi = float(((self.alpha * y).T @ kernel[:, i]).squeeze()) + float(self.bias)
                    fj = float(((self.alpha * y).T @ kernel[:, j]).squeeze()) + float(self.bias)
                    
                    Ei = fi - float(y[i].squeeze())
                    Ej = fj - float(y[j].squeeze())

                    eta = kernel[i, i] + kernel[j, j] - (2 * kernel[i, j])
                    if eta <= 0:
                        continue

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(C, C + self.alpha[j] - self.alpha[i])

                    elif y[i] == y[j]:
                        L = max(0, self.alpha[j] + self.alpha[i] - C)
                        H = min(C, self.alpha[j] + self.alpha[i])

                    old_alphaj = self.alpha[j]
                    self.alpha[j] = old_alphaj + ((y[j] * (Ei - Ej)) / eta)

                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    if self.alpha[j] < L:
                        self.alpha[j] = L

                    old_alphai = self.alpha[i]
                    self.alpha[i] = old_alphai + (y[i] * y[j] * (old_alphaj - self.alpha[j]))

                    b1 = self.bias - Ei - y[i] * (self.alpha[i] - old_alphai) * kernel[i, i] - y[j] * (self.alpha[j] - old_alphaj) * kernel[i, j]
                    b2 = self.bias - Ej - y[i] * (self.alpha[i] - old_alphai) * kernel[i, j] - y[j] * (self.alpha[j] - old_alphaj) * kernel[j, j]
                    if 0 < self.alpha[i] < C:
                        self.bias = float(b1.squeeze())
                    elif 0 < self.alpha[j] < C:
                        self.bias = float(b2.squeeze())
                    else:
                        self.bias = float(((b1 + b2) / 2).squeeze())

            if stats:
                print(f'Loss for {_} epochs: {dual_function(kernel=kernel, alphas=self.alpha, y=y)}')