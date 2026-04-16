import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import gini

class Node:
    def __init__(self, feature: int, threshold: float|str, left: 'Node' = None, right: 'Node' = None, value: int = None):
        self.feature = feature
        self.threshold = threshold
        self.gini = gini
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self):
        self.root = None
        self.max_depth = None

    def fit(self, 
            X: pd.Series|pd.DataFrame, 
            y: np.ndarray,
            max_depth: int,
            stats: bool) -> None:
        if not isinstance(X, pd.Series) and not isinstance(X, pd.DataFrame):
            return None
        if not isinstance(y, pd.Series):
            return None
        if max_depth <= 0:
            return None
        
        self.max_depth = int(max_depth)

        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        features = X.columns
        depth = 0
        n = 0
        # TODO Building tree
        




    def build_tree(self, X: pd.DataFrame, y: np.ndarray) -> None:
        pass

    def split_left_right(self,
                        feature: pd.Series, 
                        y: pd.Series, 
                        condition: str|int) -> tuple[list, list]:
        if len(feature) != len(y):
            return None
    
        n = len(y)
        left = []
        right = []
        for i in range(n):
            if type(condition) == str:
                if feature[i] == condition:
                    left.append(y[i])
                else:
                    right.append(y[i])

            else:
                if feature[i] <= condition:
                    left.append(y[i])
                else:
                    right.append(y[i])

        return (pd.Series(left), pd.Series(right), n)
    
    def best_split(self,
                   X: pd.DataFrame,
                   y: pd.Series) -> None:
        
        min_gini = []
        features = X.columns
        
        # TODO
        for feature in range(len(features)):
            if isinstance(X[feature].iloc[0], str):
                pass
            else:
                pass


    def best_threshold(self,
                       feature: pd.Series,
                       y: pd.Series) -> tuple:
        
        minimal_gini = []
         
        if isinstance(feature.iloc[0], str):
            unique = feature.unique()

            for i in range(len(unique)):
                uni = unique[i]
                
                left, right, n = self.split_left_right(feature, y, uni)

                weighted_gini = (len(left) / n * gini(left)) + (len(right) / n * gini(right))
                minimal_gini.append(weighted_gini)

            idx = np.argmin(minimal_gini)

            return unique[idx], minimal_gini[idx]
    
        else:
            thresholds = []

            sorting = pd.DataFrame({'val': feature, 'target': y})
            sorting = sorting.sort_values(by='val')
            
            feature = sorting['val']
            y = sorting['target']

            for i in range(1, len(y)):
                threshold = (feature.iloc[i - 1] + feature.iloc[i]) / 2

                left, right, n = self.split_left_right(feature, y, threshold)

                weighted_gini = (len(left) / n * gini(left)) + (len(right) / n * gini(right))
                minimal_gini.append(weighted_gini)
                thresholds.append(threshold)

            idx = np.argmin(minimal_gini)

            return thresholds[idx], minimal_gini[idx]