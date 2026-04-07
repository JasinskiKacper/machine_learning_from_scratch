import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import gini

class Node:
    def __init__(self, feature: int, threshold: float|str, left: 'Node' = None, right: 'Node' = None, value: int = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self):
        self.root = None
        self.max_depth = None

    def build_tree(self, feature: np.ndarray) -> None:
        if feature.dtype == '<3U': # non numerical feature
            pass
        else: # numercial feature
            pass


    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            max_depth: int,
            stats: bool) -> None:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        self.max_depth = int(max_depth)

        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        n = X.shape[1] # number of features
        # Building tree