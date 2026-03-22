import numpy as np
from abc import abstractmethod

class SVM:
    def __init__(self):
        self.weights = None
        self.bias = None

    @abstractmethod
    def fit(self):
        pass

    