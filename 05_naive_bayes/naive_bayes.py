import numpy as np
from abc import abstractmethod

class Naive_Bayes:
    def __init__(self):
        self.unique = None
        self.log_prior_spam = None
        self.log_prior_ham = None
        self.log_likelihood_spam = None
        self.log_likelihood_ham = None

    @abstractmethod
    def fit(self):
        pass

    def predict(self):
        pass