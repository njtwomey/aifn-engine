"""Golden values for aifn-methods/learning/survival: Cox proportional hazards with Efron ties (lifelines'
CoxPHFitter) and Breslow ties (statsmodels' PHReg), and Weibull and log-normal AFT fits (lifelines), on seeded
right-censored data with tied (rounded) times."""

from typing import cast

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, LogNormalAFTFitter, WeibullAFTFitter
from statsmodels.duration.hazard_regression import PHReg


def cases() -> dict[str, object]:
    rng = np.random.default_rng(5)
    n = 120
    x = np.column_stack([rng.integers(0, 2, n).astype(float), rng.normal(size=n)])
    beta = np.array([-0.7, 0.5])
    latent = 10 * (-np.log(rng.random(n)) / np.exp(x @ beta)) ** (1 / 1.5)
    censor = rng.exponential(30, n)
    time = np.ceil(np.minimum(latent, censor))
    event = (latent <= censor).astype(float)
    df = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "T": time, "E": event})
    efron = CoxPHFitter().fit(df, "T", "E")
    breslow = PHReg(time, x, status=event, ties="breslow").fit()
    weibull = WeibullAFTFitter().fit(df, "T", "E")
    lognormal = LogNormalAFTFitter().fit(df, "T", "E")
    wp = weibull.params_
    lp = lognormal.params_
    return {
        "x": x,
        "time": time,
        "event": event,
        "efron": {
            "coefficients": efron.params_.to_numpy(),
            "se": efron.standard_errors_.to_numpy(),
            "loglik": float(efron.log_likelihood_),
        },
        "breslow": {
            "coefficients": breslow.params,
            "se": breslow.bse,
            # statsmodels types llf as a cached property; its value is a float.
            "loglik": float(cast(float, breslow.llf)),
        },
        "weibull": {
            "intercept": float(wp["lambda_"]["Intercept"]),
            "coefficients": [float(wp["lambda_"]["x0"]), float(wp["lambda_"]["x1"])],
            "scale": float(1 / np.exp(wp["rho_"]["Intercept"])),
            "loglik": float(weibull.log_likelihood_),
        },
        "lognormal": {
            "intercept": float(lp["mu_"]["Intercept"]),
            "coefficients": [float(lp["mu_"]["x0"]), float(lp["mu_"]["x1"])],
            "scale": float(np.exp(lp["sigma_"]["Intercept"])),
            "loglik": float(lognormal.log_likelihood_),
        },
    }
