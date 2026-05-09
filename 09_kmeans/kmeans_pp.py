import numpy as np

from kmeans import Kmeans

class Kmeans_pp(Kmeans):
    def fit(self,
            X: np.ndarray,
            k: int,
            max_iter: int) -> None:
        '''
        Fit K-Means++ model to data. Initializes centroids using the K-Means++
        strategy each subsequent centroid is chosen with probability proportional
        to squared distance from the nearest already chosen centroid.
        Converges when centroids stop moving or max_iter is reached.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        k: int number of clusters
        max_iter: int maximum number of iterations
        '''
        if not isinstance(X, np.ndarray):
            return None
        k = int(k)
        max_iter = int(max_iter)

        if X.ndim == 1:
            X = X.reshape((-1, 1))

        # inicialiizng centroids
        centroids = [X[np.random.randint(0, X.shape[0])]]
        for _ in range(k - 1):
            distances = []
            for i in X:
                min_dist = min(np.linalg.norm(i - c) ** 2 for c in centroids)
                distances.append(min_dist)

            probability = distances / np.sum(distances)
            idx_by_probability = np.random.choice(len(probability), p=probability)

            centroids.append(X[idx_by_probability])
        
        for _ in range(max_iter):
            # calculating euclidean distance
            labels = []
            for i in X:
                distance = []

                for centroid in centroids:
                    diff = centroid - i
                    distance.append(np.linalg.norm(diff))

                labels.append(distance.index(np.min(distance)))
                
            labels = np.array(labels)
            # calculating new centroids
            new_centroids = []

            for cluster in range(k):
                means = []
                
                for feature in range(X.shape[1]):
                    m = X[labels == cluster][:, feature].mean()
                    means.append(m)

                new_centroids.append(means)

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        self.centroid = new_centroids
        self.labels = labels