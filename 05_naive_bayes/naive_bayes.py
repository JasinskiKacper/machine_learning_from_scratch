import numpy as np
from abc import abstractmethod, ABC

class Naive_Bayes(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass