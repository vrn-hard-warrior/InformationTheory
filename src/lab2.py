# The 2nd lab
import numpy as np
from scipy.integrate import quad
from scipy.special import rel_entr
import scipy.stats as stat

# Machine epsilon for solving zero division problem 
eps = np.finfo(np.float64).eps


def main() -> None:
    """
    Create two probability density functions with scipy.stats framework;
    compute Kullback-Leibler divergence (KL) for identical and different distributions,
    perform a verification of KL properties.
    """
    # Kullback-Leibler divergence properties (continuous case)
    p = stat.norm(loc = 0.0, scale = 1.0)
    q = stat.gamma(a = 2.99, loc = 0.0, scale = 0.5)
    
    l = 0.0001
    h = 10.0
    
    print("Continuous case:")
    print(f"Non-symmetric: {KL_continuous(p, q, low = l, high = h):.10f}",
          f"not equal {KL_continuous(q, p, low = l, high = h):.10f}")
    print(f"For equal distributions: {KL_continuous(p, p, low = l, high = h):.3f}")
    print(f"The main equality: {KL_continuous(p, q, low = l, high = h):.10f} =",
          f"{cross_entropy_continuous(p, q, low = l, high = h):.10f} -",
          f"{entropy_continuous(p, low = l, high = h):.10f}", end = '\n\n')
    
    # Kullback-Leibler divergence properties (discrete case)
    p = stat.poisson(mu = 4)
    q = stat.poisson(mu = 10)
    
    k = np.arange(0, 21, 1)
    
    print("Discrete case:")
    print(f"Non-symmetric: {KL_discrete(p, q, k):.10f}",
          f"not equal {KL_discrete(q, p, k):.10f}")
    print(f"For equal distributions: {KL_discrete(p, p, k):.3f}")
    print(f"The main equality: {KL_discrete(p, q, k):.10f} =",
          f"{cross_entropy_discrete(p, q, k):.10f} -",
          f"{entropy_discrete(p, k):.10f}", end = '\n\n')
    
    # Verification with scipy.special.rel_entr
    print("Kullback-Leibler divergence verification:", end = '\n')
    print(f"Custom: {KL_discrete(p, q, k) * np.log(2):.10f}", end = '\n')
    print(f"Scipy: {np.sum(rel_entr(p.pmf(k), q.pmf(k))):.10f}", end = '\n')
    
    return None


def KL_continuous(p, q, low: np.float64 = -10, high: np.float64 = 10) -> np.float64:
    """
    Compute Kullback-Leibler divergence for two continuous distributions.
    """
    def integrand(x: np.float32):
        if p.pdf(x) < eps or q.pdf(x) < eps:
            return 0.0
        else:
            return p.pdf(x) * np.log2(p.pdf(x) / q.pdf(x))
    
    kl_div, _ = quad(integrand, a = low, b = high)
    
    return kl_div


def KL_discrete(p, q, k: np.array) -> np.float64:
    """
    Compute Kullback-Leibler divergence for two discrete distributions.
    """
    p_pmf = p.pmf(k)
    q_pmf = q.pmf(k)
    
    p_pmf_clip = np.where(p_pmf == 0.0, eps, p_pmf)
    q_pmf_clip = np.where(q_pmf == 0.0, eps, q_pmf)
    
    kl_div = np.sum(p_pmf_clip * np.log2(p_pmf_clip / q_pmf_clip))
    
    return kl_div


def cross_entropy_continuous(p, q, low: np.float64 = -10, high: np.float64 = 10) -> np.float64:
    """
    Compute cross-entropy for two continuous distributions.
    """
    def integrand(x: np.float32):
        if q.pdf(x) < eps:
            return 0.0
        else:
            return p.pdf(x) * np.log2(q.pdf(x))
    
    cross_ent, _ = quad(integrand, a = low, b = high)
    
    return -cross_ent


def cross_entropy_discrete(p, q, k: np.array) -> np.float64:
    """
    Compute cross-entropy for two discrete distributions.
    """
    p_pmf = p.pmf(k)
    q_pmf = q.pmf(k)
    
    q_pmf_clip = np.where(q_pmf == 0.0, eps, q_pmf)
    
    cross_ent = -np.sum(p_pmf * np.log2(q_pmf_clip))
    
    return cross_ent


def entropy_continuous(p, low: np.float64 = -10, high: np.float64 = 10) -> np.float64:
    """
    Compute entropy for continuous distribution.
    """
    def integrand(x: np.float32):
        if p.pdf(x) < eps:
            return 0.0
        else:
            return p.pdf(x) * np.log2(p.pdf(x))
    
    entropy, _ = quad(integrand, a = low, b = high)
    
    return -entropy


def entropy_discrete(p, k: np.array) -> np.float64:
    """
    Compute entropy for discrete distribution.
    """
    p_pmf = p.pmf(k)
    
    p_pmf_clip = np.where(p_pmf == 0.0, eps, p_pmf)
    
    entropy = -np.sum(p_pmf_clip * np.log2(p_pmf_clip))
    
    return entropy


if __name__ == "__main__":
    main()