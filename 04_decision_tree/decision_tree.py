import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import gini

class Node:
    def __init__(self, feature: int, 
                 threshold: float|str, 
                 gin: float,
                 left: 'Node' = None, 
                 right: 'Node' = None, 
                 value: int = None):
        self.feature = feature
        self.threshold = threshold
        self.gin = gin
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self):
        self.root = None
        self.max_depth = None

    def fit(self, 
            X: pd.Series|pd.DataFrame, 
            y: pd.Series,
            max_depth: int) -> None:
        if not isinstance(X, pd.Series) and not isinstance(X, pd.DataFrame):
            return None
        if not isinstance(y, pd.Series):
            return None
        if max_depth <= 0:
            return None
        
        self.max_depth = int(max_depth)

        self.root = self.build_tree(X, y)

    def predict(self, X: pd.DataFrame) -> list:
        res = []
        for idx, row in X.iterrows():
            res.append(self.way(self.root, row))
        return res
    
    def way(self, node: Node, row: pd.Series):
        if node.value is not None:
            return node.value
        
        if isinstance(node.threshold, str):
            if row[node.feature] == node.threshold:
                return self.way(node.left, row)
            else:
                return self.way(node.right, row)
        else:
            if row[node.feature] <= node.threshold:
                return self.way(node.left, row)
            else:
                return self.way(node.right, row)

    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> None:
        if depth >= self.max_depth or len(y.unique()) == 1 or X.shape[0] <= 1:
            node = Node(feature=None, threshold=None, gin=0)
            node.value = y.value_counts().idxmax()
            return node
        
        feature, threshold, gin = self.best_split(X, y)
        node = Node(feature=feature, threshold=threshold, gin=gin)

        X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature, threshold)
        
        if len(y_left) == 0 or len(y_right) == 0:
            node.value = y.value_counts().idxmax()
            return node

        node.left = self.build_tree(X_left, y_left, depth=depth+1)
        node.right = self.build_tree(X_right, y_right, depth=depth+1)

        return node

    def split_left_right(self,
                        feature: pd.Series, 
                        y: pd.Series, 
                        condition: str|int) -> tuple[list, list]:
        if feature.shape[0] != y.shape[0]:
            return None
    
        n = len(y)
        left = []
        right = []
        for i in range(n):
            if type(condition) == str:
                if feature.iloc[i] == condition:
                    left.append(y.iloc[i])
                else:
                    right.append(y.iloc[i])

            else:
                if feature.iloc[i] <= condition:
                    left.append(y.iloc[i])
                else:
                    right.append(y.iloc[i])

        return (pd.Series(left), pd.Series(right), n)
    
    def split_dataset(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      feature_name: str,
                      threshold: str|float) -> tuple:

        X_left = []
        y_left = []
        X_right = []
        y_right = []

        for idx, row in X.iterrows():
            if type(threshold) == str:
                if row[feature_name] == threshold:
                    X_left.append(row)
                    y_left.append(y.loc[idx])
                else:
                    X_right.append(row)
                    y_right.append(y.loc[idx])

            else:
                if row[feature_name] <= threshold:
                    X_left.append(row)
                    y_left.append(y.loc[idx])
                else:
                    X_right.append(row)
                    y_right.append(y.loc[idx])

        X_left = pd.DataFrame(X_left).reset_index(drop=True)
        y_left = pd.Series(y_left).reset_index(drop=True)
        X_right = pd.DataFrame(X_right).reset_index(drop=True)
        y_right = pd.Series(y_right).reset_index(drop=True)

        return (X_left, y_left, X_right, y_right)
    
    def best_split(self,
                   X: pd.DataFrame,
                   y: pd.Series) -> tuple:
        
        min_gini = []
        thresholds = []
        features = X.columns
        
        for feature in features:
            thr, gin = self.best_threshold(X[feature], y)
            
            thresholds.append(thr)
            min_gini.append(gin)

        idx = np.argmin(min_gini)

        return (features[idx], thresholds[idx], min_gini[idx])

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