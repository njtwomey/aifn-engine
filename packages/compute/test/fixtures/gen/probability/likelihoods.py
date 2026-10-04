"""Golden values for aifn-compute/probability/likelihoods.

Pointwise log-likelihoods come from scipy.stats. Unit deviances are twice the scipy log-likelihood ratio of the
saturated model, d = 2φ(L(y; y) − L(y; μ))/w with L the log-likelihood (exponential-dispersion families). Ordinal
class probabilities come from their defining formulas over scipy's latent cdfs (logistic, normal, Gumbel-minimum).
"""

import numpy as np
import scipy.stats as st
from scipy.special import expit


def ordinal(model: str, link: str, eta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    cdf = {"logit": expit, "probit": st.norm.cdf, "cloglog": lambda z: 1 - np.exp(-np.exp(z))}[link]
    k = len(theta) + 1
    out = np.zeros((len(eta), k))
    for i, e in enumerate(eta):
        if model == "cumulative":
            c = np.concatenate([[0.0], cdf(theta - e), [1.0]])
            out[i] = np.diff(c)
        elif model == "continuation-ratio":
            surv = 1.0
            for j in range(k - 1):
                p = cdf(theta[j] - e)
                out[i, j] = surv * p
                surv *= 1 - p
            out[i, k - 1] = surv
        else:  # adjacent-category: log(p_{k+1}/p_k) = η − θ_k
            logits = np.concatenate([[0.0], np.cumsum(e - theta)])
            out[i] = np.exp(logits - np.logaddexp.reduce(logits))
    return out


def cases() -> dict[str, object]:
    phi = 0.7
    theta = 2.5
    y_cont = np.array([0.4, 1.3, 2.2, 0.9])
    mu_cont = np.array([0.5, 1.1, 2.9, 1.0])
    y_count = np.array([0.0, 1.0, 4.0, 7.0])
    mu_count = np.array([0.8, 1.5, 3.2, 6.1])
    trials = np.array([5.0, 10.0, 3.0, 8.0])
    successes = np.array([1.0, 7.0, 0.0, 8.0])
    y_prop = successes / trials
    mu_prop = np.array([0.3, 0.55, 0.2, 0.9])

    def famcase(
        logpdf: object, variance: np.ndarray, y: np.ndarray, mu: np.ndarray, dispersion: float, w: np.ndarray
    ) -> dict:
        lp = logpdf(y, mu)  # type: ignore[operator]
        saturated = logpdf(y, y)  # type: ignore[operator]
        return {
            "y": y,
            "mu": mu,
            "weights": w,
            "logProb": lp,
            "variance": variance,
            "unitDeviance": 2 * dispersion * (saturated - lp) / w,
        }

    ones = np.ones(4)
    eta = np.array([-1.0, 0.0, 0.8, 2.5])
    thresholds = np.array([-0.5, 0.4, 1.7])
    ordinals = {
        f"{model}-{link}": ordinal(model, link, eta, thresholds)
        for model in ["cumulative", "continuation-ratio"]
        for link in ["logit", "probit", "cloglog"]
    }
    ordinals["adjacent-category-logit"] = ordinal("adjacent-category", "logit", eta, thresholds)
    return {
        "phi": phi,
        "theta": theta,
        "gaussian": famcase(lambda y, m: st.norm.logpdf(y, m, np.sqrt(phi)), np.ones(4), y_cont, mu_cont, phi, ones),
        "gamma": famcase(
            lambda y, m: st.gamma.logpdf(y, 1 / phi, scale=m * phi), mu_cont**2, y_cont, mu_cont, phi, ones
        ),
        "inverseGaussian": famcase(
            lambda y, m: st.invgauss.logpdf(y, m * phi, scale=1 / phi), mu_cont**3, y_cont, mu_cont, phi, ones
        ),
        "poisson": famcase(lambda y, m: st.poisson.logpmf(y, m), mu_count, y_count, mu_count, 1.0, ones),
        "negativeBinomial": famcase(
            lambda y, m: st.nbinom.logpmf(y, theta, theta / (theta + m)),
            mu_count + mu_count**2 / theta,
            y_count,
            mu_count,
            1.0,
            ones,
        ),
        "binomial": famcase(
            lambda y, m: st.binom.logpmf(y * trials, trials, m), mu_prop * (1 - mu_prop), y_prop, mu_prop, 1.0, trials
        ),
        "ordinal": {"eta": eta, "thresholds": thresholds, "probabilities": ordinals},
    }
