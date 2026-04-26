import numpy as np
import pandas as pd
from naive_bayes import Naive_Bayes

class Multinomial_nb(Naive_Bayes):
    def vectorizer(self, X: pd.Series) -> np.ndarray:
        unique = {}
        col = 0

        for i in X:
            i = i.lower()
            words = i.split()
            for word in words:
                if word not in unique:
                    unique[word] = col
                    col += 1

        feature_matrix = np.ones(shape=(X.shape[0], len(unique)))
        for idx, i in enumerate(X):
            i = i.lower()
            words = i.split()
            for word in words:
                feature_matrix[idx][unique[word]] += 1


        return feature_matrix, unique
    
    def fit(self, 
            X: pd.Series,
            y: pd.Series) -> None:
        '''
        
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples, 1)
        '''
        if not isinstance(X, pd.Series) or not isinstance(y, pd.Series):
            return None
        
        n = len(y)
        spam = 0

        for res in y:
            if res == 'spam':
                spam += 1
        ham = n - spam

        P_spam = spam / n
        P_ham = ham / n

        _, unique = self.vectorizer(X)
        n_unique = len(unique)
        spam_values = np.ones(shape=(n_unique, 1))
        ham_values = np.ones(shape=(n_unique, 1))
        for sms, res in zip(X, y):
            words = sms.lower().split()
            for word in words:
                if res == 'spam':
                    spam_values[unique[word]] += 1
                elif res == 'ham':
                    ham_values[unique[word]] += 1

        P_words_spam = []
        P_words_ham = []
        for i in range(len(unique)):
            prob_spam = spam_values[i] / spam_values.sum()
            prob_ham = ham_values[i] / ham_values.sum()
        
            P_words_spam.append(prob_spam)
            P_words_ham.append(prob_ham)

        self.unique = unique
        self.log_prior_spam = np.log(P_spam)
        self.log_prior_ham = np.log(P_ham)
        self.log_likelihood_spam = np.log(np.array(P_words_spam))
        self.log_likelihood_ham = np.log(np.array(P_words_ham))