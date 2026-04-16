import numpy as np
import pandas as pd

def gini(y: pd.Series) -> float:
    n = len(y)
    
    counts = y.value_counts()
    uniques = y.unique()
    gini_sum = 0
    for unique in uniques:
        gini_sum += (counts[unique] / n) ** 2

    return 1 - gini_sum