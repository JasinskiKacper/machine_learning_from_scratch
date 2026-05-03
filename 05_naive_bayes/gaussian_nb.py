import numpy as np
import pandas as pd
from naive_bayes import Naive_Bayes

class Gaussian_nb(Naive_Bayes):
    def __init__(self):
        pass
    
    def vectorizer(self, X: pd.Series) -> np.ndarray:
        pass
    
    def fit(self, 
            X: pd.Series,
            y: pd.Series) -> None:
        pass

    def predict(self, X: str|pd.Series) -> str|list:
        pass

    