import numpy as np
from abc import abstractmethod, ABC

class PCA(ABC):
    def __init__(self):
        self.explained_ratio = None

    @abstractmethod
    def fit_transform(self):
        pass