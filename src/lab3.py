# The 3rd lab
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from typing import Tuple
from lab1 import rng_init

# For annotations
from numpy.typing import NDArray


def main() -> None:
    """
    Three signals creation and application of the mixing matrix.
    """
    fs = 1e4                           # Sample rate [Hz]
    N = 65000                          # Training dataset length
    rng = rng_init("PCG64")            # Random Number Generator initialization
    
    t = np.arange(N, dtype = np.float64)
    s0 = np.array(list(map(signal_0, t)))
    s1 = np.array(list(map(signal_1, t)))
    s2 = np.array(list(map(signal_2, t, np.repeat(rng, N))))
    
    # Visualize signal
    plotting(t, s0, fs, (-0.2, 0.2), 0.1)
    
    # Mixing matrix
    A = np.array([[0.56, 0.79, -0.37], [-0.75, 0.65, 0.86], [0.17, 0.32, -0.48]], dtype = np.float64)
    
    # Linear mixing
    U = np.vstack((s0, s1, s2))
    X = mixer(U, A)
    
    # Visualize mix
    plotting(t, X[0, :], fs, (-0.8, 0.8), 0.01)
    
    # Demixing matrix
    W = np.array([[4.1191, -1.7879, -6.3765], [-10.1932, -9.8141, -9.7259], [0.2222, 0.0294, -0.6213]], dtype = np.float64)
    
    # Linear demixing
    Y = mixer(X, W)
    
    # Visualize recovered signal
    plotting(t, Y[0, :], fs, (-0.4, 0.4), 0.1)
    
    return None


def mixer(U: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.matmul(A, U)


def plotting(t: np.array, x: np.array, fs: float = 1e4, ylim: Tuple[float, float] = (-0.2, 0.2), xlim: float = 0.01) -> None:
    plt.plot(t / fs, x, 'k-', alpha = 1.0)
    plt.xlim((0.0, xlim))
    plt.ylim(ylim)
    plt.grid()
    plt.show()
    
    return None


def global_rejection_index(P: NDArray) -> float:
    P = np.sum(np.sum(np.abs(P) / np.max(np.abs(P), axis = 1, keepdims = True), axis = 1) - 1) + \
        np.sum(np.sum(np.abs(P) / np.max(np.abs(P), axis = 0, keepdims = True), axis = 0) - 1)
    
    return P


def signal_0(n: int = 1, A: float = 0.1, f: float = 100.0, f_d: float = 40.0, fs: float = 1e4) -> float:
    return A * np.sin(2 * np.pi * f / fs * n) * np.cos(2 * np.pi * f_d / fs * n)


def signal_1(n: int = 1, A: float = 0.1, A_d: float = 9.0, f: float = 500.0, f_d: float = 40.0, fs: float = 1e4) -> float:
    return A * heaviside_fun(np.sin(2 * np.pi * f / fs * n + A_d * np.cos(2 * np.pi * f_d / fs * n)))


def signal_2(n: int, rng: rnd._generator.Generator) -> float:
    return rng.uniform(low = -1.0, high = 1.0)


def signal_3(n: int = 1, A: float = 0.1, f0: float = -20.0, F_max: float = 20.0, N: int = 1000, fs: float = 1e4) -> float:
    return A * np.cos(2 * np.pi * (f0 / fs * n + (F_max - f0) / (fs * N) * n ** 2))


def signal_4(n: int, rng: rnd._generator.Generator) -> float:
    return rng.normal(loc = 0.0, scale = 0.1)


def signal_5(n: int, A: float = 0.1, f0: float = 17.0, f1: float = 19.0, fs: float = 1e4) -> float:
    return A * np.sin(2 * np.pi * f0 / fs * n + np.pi / 6) + A * np.cos(2 * np.pi * f1 / fs * n + np.pi / 12)


def signal_6(n: int = 1, A: float = 0.1, A_d: float = 9.0, f: float = 300.0, f_d: float = 40.0, fs: float = 1e4) -> float:
    return A * np.sin(2 * np.pi * f / fs * n + A_d * heaviside_fun(np.cos(2 * np.pi * f_d / fs * n)))


def signal_7(n: int = 1, A: float = 0.1, A_d: float = 9.0, f: float = 500.0, f_d: float = 40.0, fs: float = 1e4) -> float:
    return A * np.sin(2 * np.pi * f / fs * n + A_d * np.cos(2 * np.pi * f_d / fs * n))


def heaviside_fun(x: float) -> float:
    if x < 0:
        return 1.0
    else:
        return -1.0


if __name__ == '__main__':
    main()