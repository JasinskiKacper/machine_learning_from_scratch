import numpy as np

from principal_component_analysis import PCA

class PCA_eigenvectors(PCA):
    def fit_transform(self,
                      X: np.ndarray,
                      k: int):
        '''
        Fit PCA model using eigendecomposition of the covariance matrix
        and transform data into k-dimensional space.
        Computes the covariance matrix of centered data, extracts eigenvectors
        sorted by explained variance, and projects data onto the top k components.

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

        # covarianvce matrix
        n = X.shape[0]

        cov_matrix = (1 / (n - 1)) * X_cen.T @ X_cen

        # eigen values and vectors
        eig_val, eig_vec = np.linalg.eig(cov_matrix)
        
        eigen = list(zip(eig_val, eig_vec.T))
        eigen.sort(key=lambda x: x[0], reverse=True)

        # transformation matrix
        W = []
        for i in range(k):
            W.append(eigen[i][1])
        
        W = np.array(W).T

        # data projection
        X_pca = X_cen @ W

        # calculating ratios
        eig_vals_sorted = np.array([val for val, _ in eigen])
        self.explained_ratio = eig_vals_sorted / eig_vals_sorted.sum()

        return X_pca