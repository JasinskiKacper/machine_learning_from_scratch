import numpy as np

class K_nearest_neighbors:
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Store the training data. KNN is a lazy learner, so no actual 
        training happens here; it simply memorizes the dataset.
        
        Parameters
        -----------
        X : np.ndarray
            Training features (n_samples, n_features).
        y : np.ndarray
            Target values (n_samples, 1).
        '''
        self.X = X
        self.y = y

    def distance(self, Xi: float|np.ndarray) -> np.ndarray:
        '''
        Calculate the Euclidean distance between a single query point (Xi) 
        and all points stored in the training set (self.X).

        Parameters
        ----------
        Xi : float or np.ndarray
            A single data point to compare against the training set.
        '''
        diff = self.X - Xi
        if diff.ndim > 1:
            sum_dist = np.sum(diff ** 2, axis=1)
        else:
            sum_dist = diff ** 2

        return np.sqrt(sum_dist)
    
    def predict_regressor(self, X: float|np.ndarray, k: int) -> float|np.ndarray:
        '''
        Predict the target value for given input X by calculating 
        the mean of its k-nearest neighbors in the training set.

        Parameters
        -----------
        X : float or np.ndarray
            Input features to predict.
        k : int
            The number of nearest neighbors to include in the average.
        '''
        if not isinstance(X, float) and not isinstance(X, np.ndarray):
            return None
        k = int(k)

        if X.ndim == 1:
            X = np.reshape(X, shape=(-1, 1))

        y_pred = np.zeros(shape=(X.shape[0], 1))

        for i in range(X.shape[0]):

            dist = self.distance(X[i])
            X_distance = np.c_[self.X, self.y, dist]
            X_sorted = X_distance[X_distance[:, -1].argsort()]

            y_pred[i] = np.sum(X_sorted[: k, -2]) / k

        return y_pred
