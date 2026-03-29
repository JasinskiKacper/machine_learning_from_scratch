import numpy as np

def hinge_loss(X: np.ndarray,
                y: np.ndarray,
                weights: np.ndarray, 
                bias: float|int,
                ) -> float:
    total_loss = 0
    size = y.shape[0]
    for i in range(size):
        margin = y[i] * ((X[i] @ weights) + bias)
        total_loss += max(0, 1 - margin)

    return total_loss / size

def hinge_loss_C(X: np.ndarray,
                y: np.ndarray,
                weights: np.ndarray, 
                bias: float|int,
                C: float|int
                ) -> float:
    total_loss = 0
    size = y.shape[0]
    for i in range(size):
        margin = y[i] * ((X[i] @ weights) + bias)
        margin = margin.item()
        total_loss += max(0, 1 - margin)
    
    regularization = 0.5 * np.sum(weights**2)
    
    return  regularization + C * total_loss / size