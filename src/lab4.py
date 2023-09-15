# The 4th lab
import numpy as np
import numpy.random as rnd

from lab1 import rng_init
from lab3 import signal_0, signal_1, signal_2, plotting, mixer

# For annotations
from numpy.typing import NDArray
from typing import Tuple


def main() -> None:
    """
    Three signals creation, application of the mixing matrix and making signals reconstruction.
    """
    fs = 1e4                           # Sample rate [Hz]
    alpha = 0.1                        # Learning rate
    eps = 1e-14                        # Convergence tolerance
    N = 65000                          # Training dataset length
    epochs = 10                        # Training epochs
    batch_size = 100                   # Batch size for packet-learning
    case = 'fix'                       # Two cases for activation function
    rng = rng_init("PCG64")            # Random Number Generator initialization
    
    t = np.arange(N, dtype = np.float64)
    s0 = np.array(list(map(signal_0, t)))
    s1 = np.array(list(map(signal_1, t)))
    s2 = np.array(list(map(signal_2, t, np.repeat(rng, N))))
    
    # Visualize signal
    plotting(t, s2, fs, (-1.0, 1.0), 0.01)
    
    # Mixing matrix creation
    A = np.array([[0.56, 0.79, -0.37], [-0.75, 0.65, 0.86], [0.17, 0.32, -0.48]], dtype = np.float64)
    
    # Linear mixing
    U = np.vstack((s0, s1, s2))
    X = mixer(U, A)
    
    # Visualize mix
    plotting(t, X[0, :], fs, (-0.8, 0.8), 0.3)
    
    # Whitening and centering
    X_preprocessed, W_preprocessed = preprocessing(X)
    
    # Demixing matrix creation
    W = rng.uniform(low = 0.0, high = 0.05, size = (3, 3))
    
    # ICA recovering by S. Haykin
    # W = ICA_recovering_H(X, W, rng, alpha, batch_size, epochs, case)
    
    # ICA recovering by A. Hyvarinen and E. Oja
    W = ICA_recovering_O(X_preprocessed, W, eps)
    
    # Linear demixing
    Y = mixer(X_preprocessed, W)
    
    plotting(t, -Y[1, :], fs, (-2.2, 2.2), 0.01)
    
    return None


def ICA_recovering_H(X: NDArray,
                   W: NDArray,
                   rng: rnd._generator.Generator,
                   alpha: float = 0.1,
                   batch_size = 100,
                   epochs: int = 300,
                   case: str = 'unfix') -> NDArray:
    """
    Recovering the mix of signals with Independent Component Analysis (see S. Haykin "Neural Networks").
    """
    N_samples = int(np.floor(X.shape[1] / batch_size))
    
    if case == "unfix":
        phi = phi_haykin
    elif case == "fix":
        phi = phi_amari
    else:
        return None
    
    for i in range(epochs):
        X_rnd = rng.permutation(X, axis = 1)
        
        for j in range(N_samples):
            X_batch = X_rnd[:, j * batch_size: (j + 1) * batch_size]
            
            # Linear demixing
            Y = mixer(X_batch, W)
            PHI = phi(Y)
            
            W += alpha * np.matmul(np.eye(3, dtype = float) - 1 / batch_size * np.matmul(PHI, Y.T), W)
    
    return W


def phi_haykin(y: float) -> float:
    """
    Activation function from S. Haykin "Neural Networks".
    """
    return 0.5 * np.power(y, 5) + 2 / 3 * np.power(y, 7) + 15 / 2 * np.power(y, 9) + \
        2 / 15 * np.power(y, 11) - 112 / 3 * np.power(y, 13) + 128 * np.power(y, 15) - \
        512 / 3 * np.power(y, 17)


def phi_amari(y: float) -> float:
    """
    Activation function from Amari S., Cichocki A., Yang H. H.
    "A New Learning Algorithm for Blind Signal Separation"
    """
    return 29 / 4 * np.power(y, 3) - 47 / 4 * np.power(y, 5) - 14 / 3 * np.power(y, 7) + \
        25 / 4 * np.power(y, 9) + 3 / 4 * np.power(y, 11)


def ICA_recovering_O(X: NDArray, W: NDArray, eps: float = 1e-15) -> NDArray:
    """
    Recovering the mix of signals with ideas from
    Independent Component Analysis: Algorithms and Applications - A. Hyvarinen and E. Oja.
    """
    N = X.shape[0]
    N_data = X.shape[1]
    
    for i in range(N):
        w_old = np.zeros(N)
        w_new = W[i, :]
        
        while np.abs(np.matmul(w_new.T, w_old)) < (1.0 - eps):
            w_old = w_new
            
            y = np.matmul(w_old, X)
            w_new = np.matmul(X, g(y)) / N_data - np.mean(g_der(y)) * w_old
            
            # Using Gram-Schmidt ortogonalization to decorrelate the vectors
            for j in range(i):
                w_new -= np.matmul(w_new, W[j, :].T) * W[j, :]
            
            w_new /= np.linalg.norm(w_new, ord = 2)
        
        W[i, :] = w_new
    
    return W


def preprocessing(X: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Whitening and centering of mixed signals.
    """
    X = (X - np.mean(X, axis = 1, keepdims = True)) / np.std(X, axis = 1, keepdims = True)
    X_cov = np.corrcoef(X, rowvar = True)
    
    eigval, eigvec = np.linalg.eigh(X_cov)
    
    W_preprocessed = np.matmul(np.matmul(eigvec, np.diag(np.sqrt(1.0 / eigval))), eigvec.T)
    X_preprocessed = np.matmul(W_preprocessed, X)
    
    return X_preprocessed, W_preprocessed


def g(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def g_der(x: float) -> float:
    return g(x) * (1.0 - g(x))


if __name__ == '__main__':
    main()