import numpy as np
import pandas as pd
from naive_bayes import Naive_Bayes

class Gaussian_nb(Naive_Bayes):
    def __init__(self):
        self.unique = None
        self.class_probability = None
        self.mean = None
        self.std = None

    def fit(self, 
            X: pd.Series|pd.DataFrame,
            y: pd.Series) -> None:
        '''
        Trains the Gaussian Naive Bayes model by 
        calculating class probability, means and std.

        Parameters
        ----------
        X : pd.Series
        y : pd.Series
        '''
        if not isinstance(X, pd.Series) and not isinstance(X, pd.DataFrame):
            return None
        if not isinstance(y, pd.Series):
            return None
        
        self.unique = list(y.unique())
        n = y.shape[0]
        self.class_probability = [0 for _ in range(len(self.unique))]
        for i in y:
            idx = self.unique.index(i)
            self.class_probability[idx] += 1

        self.class_probability = [i / n for i in self.class_probability]

        self.mean = {}
        self.std = {}
        features = list(X.columns)

        for u in self.unique:
            mean = []
            std = []
            X_u = X[y == u]
            for feature in features:
                mean.append(X_u[feature].mean())
                std.append(X_u[feature].std())

            self.mean[u] = mean
            self.std[u] = std

    def predict(self, X: pd.DataFrame) -> list:
        '''
        Calculates the log probabilities for each class 
        using the Gaussian probability density function.

        Parameters
        ----------
        X : pd.DataFrame The input features to be classified.
        '''
        result = []
        n = X.shape[0]
        
        log_likelihood_sum = 0
        for row in range(n):
            choice = [np.log(prob) for prob in self.class_probability]
            for i in range(len(self.class_probability)):
                for idx, feature in enumerate(list(X.columns)):
                    log_likelihood_sum += np.log(self.density(X[feature][row], 
                                                self.mean[self.unique[i]][idx],
                                                self.std[self.unique[i]][idx]))   

                choice[i] = choice[i] + log_likelihood_sum
                log_likelihood_sum = 0
            result.append(choice)
            
        return result
    
    def predict_class(self, X: pd.Series) -> list:
        '''
        Predicts the class labels for the provided input data by 
        selecting the class with the highest log probability.

        Parameters
        ----------
        X : pd.DataFrame The input features to be classified.
        '''
        result = self.predict(X)

        classes = []
        for i in range(len(result)):
            idx = result[i].index(max(result[i]))
            classes.append(self.unique[idx])

        return classes


    def density(self, x: float, mean: float, std: float) -> float:
        '''
        Calculates the Gaussian Normal probability density function 
        value for a given feature value.

        Parameters
        ----------
        x : float The value of the feature for which the density is calculated.
        mean : float The mean of the feature for a specific class.
        std : float The standard deviation of the feature for a specific class.
        '''
        return (1 / (np.sqrt(2 * np.pi * (std ** 2)))) * np.exp( - ((x - mean) ** 2) / (2 * (std ** 2)))