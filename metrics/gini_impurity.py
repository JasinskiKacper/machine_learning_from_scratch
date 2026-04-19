import numpy as np
import pandas as pd

def gini(y: pd.Series) -> float:
    if len(y) == 0:
        return 0.0
    
    n = len(y)
    counts = y.value_counts()
    uniques = y.unique()
    
    gini_sum = 0
    for unique in uniques:
        p = (y == unique).sum() / n
        gini_sum += p ** 2

    return 1 - gini_sum