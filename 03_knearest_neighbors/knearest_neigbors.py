import numpy as np

class K_nearest_neighbors:
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def distance(self, X: float|np.ndarray) -> np.ndarray:
        diff = self.X - X
        sum_dist = np.sum(diff ** 2, axis=1)

        return np.sqrt(sum_dist)
    
    def predict_regressor(self, X: float|np.ndarray, k: int) -> float|np.ndarray:
        '''
        Predicting X value by calculating mean of k nearest neighbors

        Parameters
        -----------
        X: float or np.ndarray ...
        k: amount of nearest neighbors ...
        '''
        if not isinstance(X, float) and not isinstance(X, np.ndarray):
            return None
        k = int(k)

        if X.ndim() == 1:
            X = np.reshape(X, shape=(-1, 1))


