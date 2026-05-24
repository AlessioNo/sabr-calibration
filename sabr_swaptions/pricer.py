"""
pricer.py
---------
Pricing and Greeks for swaptions under the Black (1976) model.

Used in conjunction with a calibrated SABRSurface to obtain the
implied vol at any strike before pricing.

Public API
----------
  black_price(F, K, sigma, T, option_type)
      → float  (undiscounted, per unit of annuity)

  swaption_price(F, K, sigma, T, annuity, notional, option_type)
      → float  (euros)

  black_greeks(F, K, sigma, T, annuity, notional, option_type)
      → dict   {delta_bp, vega_1pct, gamma_bp2, vanna, volga}
"""

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
#  Black (1976) formula                                                         #
# --------------------------------------------------------------------------- #

def black_price(
    F: float,
    K: float,
    sigma: float,
    T: float,
    option_type: str = "payer",
) -> float:
    """
    Black (1976) formula for a swaption.

    Returns the price as a fraction of the annuity (multiply by A × Notional
    to get the full euro price).

    Parameters
    ----------
    F           : forward swap rate
    K           : strike
    sigma       : Black implied vol (lognormal)
    T           : option expiry in years
    option_type : "payer"    → right to pay fixed  (call analogy)
                  "receveur" → right to receive fixed (put analogy)
    """
    if sigma <= 0 or T <= 0:
        if option_type == "payer":
            return max(F - K, 0.0)
        else:
            return max(K - F, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type == "payer":
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    else:  # receveur
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)


# --------------------------------------------------------------------------- #
#  Full swaption price (euros)                                                  #
# --------------------------------------------------------------------------- #

def swaption_price(
    F: float,
    K: float,
    sigma: float,
    T: float,
    annuity: float,
    notional: float,
    option_type: str = "payer",
) -> float:
    """
    Full swaption price in euros.

        Price = Notional × Annuity × Black(F, K, σ, T)

    Parameters
    ----------
    F           : forward swap rate
    K           : strike
    sigma       : Black implied vol
    T           : option expiry in years
    annuity     : PV01 / annuity factor  A(T_start, tenor)
    notional    : notional in euros
    option_type : "payer" or "receveur"
    """
    return notional * annuity * black_price(F, K, sigma, T, option_type)


# --------------------------------------------------------------------------- #
#  Analytical Greeks (Black model)                                              #
# --------------------------------------------------------------------------- #

def black_greeks(
    F: float,
    K: float,
    sigma: float,
    T: float,
    annuity: float,
    notional: float,
    option_type: str = "payer",
) -> dict:
    """
    Analytical Greeks of the swaption under the Black model.

    Returns
    -------
    dict with keys:
        delta_bp   : ∂Price/∂F  in €/bp  (impact of +1bp move in forward rate)
        vega_1pct  : ∂Price/∂σ  in €     (impact of +1% absolute vol move)
        gamma_bp2  : ∂²Price/∂F² in €    (impact of +1bp² convexity)
        vanna       : ∂²Price/∂F∂σ in €  (cross-sensitivity delta/vol)
        volga       : ∂²Price/∂σ² in €   (vol convexity, for +1% vol move²)
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    phi = norm.pdf(d1)  # standard normal density at d1

    sign = 1 if option_type == "payer" else -1

    # ── Delta ──────────────────────────────────────────────────────────────── #
    # Raw delta: ∂(undiscounted price per annuity) / ∂F
    delta_raw = sign * norm.cdf(sign * d1)
    delta_eur = notional * annuity * delta_raw   # €/unit-of-F
    delta_bp  = delta_eur * 1e-4                 # €/bp  (1bp = 0.0001)

    # ── Vega ───────────────────────────────────────────────────────────────── #
    # Expressed per +1% absolute vol move (Δσ = 0.01)
    vega_raw  = F * phi * sqrtT
    vega_1pct = notional * annuity * vega_raw * 0.01

    # ── Gamma ──────────────────────────────────────────────────────────────── #
    # Per +1bp² of forward rate
    gamma_raw = phi / (F * sigma * sqrtT)
    gamma_bp2 = notional * annuity * gamma_raw * (1e-4) ** 2

    # ── Vanna : ∂²/(∂F ∂σ) ────────────────────────────────────────────────── #
    # Per +1bp of F and +1% of vol
    vanna_raw = -phi * d2 / sigma
    vanna     = notional * annuity * vanna_raw * 1e-4 * 0.01

    # ── Volga : ∂²/∂σ² ────────────────────────────────────────────────────── #
    # Per (+1%)² of vol
    volga_raw = F * phi * sqrtT * d1 * d2 / sigma
    volga     = notional * annuity * volga_raw * (0.01) ** 2

    return {
        "delta_bp":  delta_bp,
        "vega_1pct": vega_1pct,
        "gamma_bp2": gamma_bp2,
        "vanna":     vanna,
        "volga":     volga,
    }
