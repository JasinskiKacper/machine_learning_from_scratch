import numpy as np

from kmeans import Kmeans

class Standard_Kmeans(Kmeans):
    def fit(self,
            X: np.ndarray,
            k: int,
            max_iter: int) -> None:
        '''
        Fit standard K-Means model to data using Lloyd s algorithm.
        Initializes centroids randomly, then alternates between assigning
        each point to the nearest centroid and updating centroids to the
        mean of assigned points. Converges when centroids stop moving
        or max_iter is reached.

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

        # inicialiizng random centroids
        centroids = []
        idx = np.arange(0, X.shape[0], 1)
        for _ in range(k):
            np.random.shuffle(idx)

            temp = [X.T[i, idx[i]] for i in range(X.shape[1])]
            centroids.append(temp)

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