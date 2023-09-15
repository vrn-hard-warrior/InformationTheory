# The 1st lab
import secrets
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import numpy.random as rnd
from IPython.core.pylabtools import figsize
from sklearn.neighbors import KernelDensity


def main() -> None:
    """
    Create random number generator with numpy.random framework
    (Permuted Congruential Generator - PCG XSL RR 128/64; Mersenne Twister - MT19937);
    sample certain distribution and estimate probability density function:
    - Using histogram;
    - Using kernel density estimation;
    Compare it with analytical dependencies.
    """
    rng = rng_init("PCG64")
    
    data = rng.beta(a = 2, b = 5, size = 10000)
    x = np.linspace(0, 1, 100, endpoint = True)
    
    pdf_KDE = fitKDE(data, h = 0.02, kernel = "gaussian", x = x)
    
    figsize(12, 6)
    plt.plot(x, pdf_KDE, color = 'darkorange', alpha = 0.8, lw = 2.5)
    plt.plot(x, stat.beta.pdf(x, a = 2, b = 5), color = 'navy', alpha = 0.8, lw = 2.5)
    plt.hist(data, bins = 20, range = (0, 1), density = True, histtype = "barstacked", 
             color = "lightblue")
    plt.xlim((-0.05, 1.05))
    plt.ylim((0, 2.6))
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('P(x)', fontsize = 16)
    plt.legend(["KDE", "PDF", "HIST"], fontsize = 15)
    plt.title("Beta distribution", fontsize = 18)
    plt.grid()
    
    return None


def rng_init(rng_name: str = "PCG64") -> rnd.Generator:
    """
    Creating random number generator.
    """
    init = secrets.randbits(128)
    ss = rnd.SeedSequence(init)
    
    if rng_name.lower() == "pcg64":
        bit_gen = rnd.PCG64(ss)
    elif rng_name.lower() == "mt19937":
        bit_gen = rnd.MT19937(ss)
    else:
        raise TypeError("Wrong random number generator name!")
        
    rng = rnd.Generator(bit_gen)
    
    return rng


def fitKDE(data: np.ndarray, h: np.float64 = 0.25, kernel: str = "epanechnikov",
           x: np.ndarray = None) -> np.array:
    """
    Fit kernel to a array of data, and derive the prob of obs x is the array of values
    on which the fit KDE will be evaluated. It is the empirical PDF.
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    kde = KernelDensity(kernel = kernel, bandwidth = h).fit(data)
    
    if x is None:
        x = np.unique(data).reshape(-1, 1)
        
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    log_prob = kde.score_samples(x)
    
    pdf = np.array(np.exp(log_prob))
    
    return pdf


if __name__ == "__main__":
    main()