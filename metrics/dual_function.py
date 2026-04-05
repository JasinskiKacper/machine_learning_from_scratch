import numpy as np

def dual_function(kernel: np.ndarray,
                  alphas: np.ndarray,
                  y: np.ndarray) -> float:
    sum_alphas = np.sum(alphas)

    ay = alphas * y

    margin = 0.5 * np.dot(ay.T, np.dot(kernel, ay))

    return sum_alphas - margin