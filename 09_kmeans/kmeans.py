import numpy as np
from abc import abstractmethod

class Kmeans:
    def __init__(self):
        self.centroid = None
        self.labels = None

    @abstractmethod
    def fit(self):
        pass

    def predict(self, X: np.ndarray) -> list:
        distances = []

        if X.ndim == 1:
            for j in range(len(self.centroid)):
                diff = X - self.centroid[j]
                distance = np.linalg.norm(diff)
                distances.append(distance)

            return distances.index(np.min(distances))

        label = []
        for i in X:
            distances = []
            
            for j in range(len(self.centroid)):
                diff = i - self.centroid[j]
                distance = np.linalg.norm(diff)

                distances.append(distance)
            
            label.append(distances.index(np.min(distances)))
        
        return label
                
                    


        