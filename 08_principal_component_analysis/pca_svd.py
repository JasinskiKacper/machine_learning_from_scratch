import numpy as np

from principal_component_analysis import PCA

class PCA_SVD(PCA):
    def fit_transform(self,
                      X: np.ndarray,
                      k: int):
        '''
        Fit PCA model using Singular Value Decomposition (SVD)
        and transform data into k-dimensional space.
        More numerically stable than eigendecomposition.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features) Input data to fit and transform.
        k: int Number of principal components to keep.
        '''
        if not isinstance(X, np.ndarray):
            return None
        k = int(k)

        # centering data
        feature_mean = np.mean(X, axis=0)
        X_cen = X - feature_mean

        # singular value decomposition
        U, S, Vt = np.linalg.svd(X_cen)

        X_pca = U[:, :k] * S[:k]
        
        # calculating ratios
        n = X.shape[0]
        variances = (S ** 2 / (n - 1))
        self.explained_ratio = variances / np.sum(variances)

        return X_pca