import numpy as np
from abc import abstractmethod

class PCA:
    def __init__(self):
        self.explained_ratio = None

    @abstractmethod
    def fit_transform(self):
        pass