import numpy as np
from SVM import Support_Vector_Machine

class Kernel_svm_RBF(Support_Vector_Machine):
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_iters: int,
            lr: float,
            stats: bool,
            C: float) -> None:
        '''
        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples, 1)
        max_iters: int numbers of maximum iterations 
        lr: float learning rate that dictate how much gradients influence
            weights and bias
        stats: bool showing loss every iteration
        C: float penalty parameter - high C = narrow margin, low C = wide margin
        '''
        pass