import numpy as np
import pandas as pd
from naive_bayes import Naive_Bayes

class Multinomial_nb(Naive_Bayes):
    def __init__(self):
        self.unique = None
        self.log_prior_spam = None
        self.log_prior_ham = None
        self.log_likelihood_spam = None
        self.log_likelihood_ham = None
    
    def vectorizer(self, X: pd.Series) -> np.ndarray:
        '''
        Creates a vocabulary of unique words and encodes
        messages into a bag of words feature matrix.

        Parameters
        ----------
        X : pd.Series of strings (SMS messages) to be processed.
        '''
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
        Trains the Multinomial Naive Bayes model by 
        calculating log-priors and log-likelihoods.

        Parameters
        ----------
        X : pd.Series consisting of SMS messages.
        y : pd.Series corresponding to each message 'spam' or 'ham'.
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

    def predict(self, X: str|pd.Series) -> str|list:
        '''
        Predicts the class label for the given input.

        Parameters
        ----------
        X : str or pd.Series of strings to classify.
        '''
        if isinstance(X, str):
            return self._predict(X)
        elif isinstance(X, pd.Series):
            return [self._predict(sms) for sms in X]

    def _predict(self, X: str) -> str:
        '''
        Helper method to perform prediction on a single string.

        Parameters
        ----------
        X : str SMS message to be classified.
        '''
        result_spam = self.log_prior_spam
        result_ham = self.log_prior_ham

        words = X.lower().split()

        for word in words:
            if word in self.unique:
                result_ham += self.log_likelihood_ham[self.unique[word]]
                result_spam += self.log_likelihood_spam[self.unique[word]]

        if result_spam > result_ham:
            return 'spam'
        else:
            return 'ham'