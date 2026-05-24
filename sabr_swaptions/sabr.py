"""
sabr.py
-------
SABR model implementation following Hagan et al. (2002):
    "Managing Smile Risk", Wilmott Magazine.

Provides:
  - sabr_vol()          : implied vol formula (scalar and vectorised)
  - calibrate_sabr()    : least-squares calibration of (alpha, rho, nu) for fixed beta
  - SABRSurface         : calibrate the full swaption surface and store results
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional
import warnings


# --------------------------------------------------------------------------- #
#  Hagan et al. (2002) SABR implied vol formula                                #
# --------------------------------------------------------------------------- #

def sabr_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    atm_tolerance: float = 1e-7,
) -> float:
    """
    SABR implied (normal) lognormal vol via the Hagan 2002 expansion.

    Parameters
    ----------
    F     : forward rate (par swap rate)
    K     : strike
    T     : option expiry in years
    alpha : initial vol parameter (> 0)
    beta  : CEV exponent in [0, 1]; typically fixed (0 or 0.5 or 1)
    rho   : correlation ∈ (-1, 1)
    nu    : vol-of-vol (≥ 0)

    Returns
    -------
    sigma_B : Black (lognormal) implied vol
    """
    if T <= 0:
        return 0.0

    FK   = F * K
    FKm  = FK ** ((1.0 - beta) / 2.0)
    log_FK = np.log(F / K) if abs(F / K - 1.0) > atm_tolerance else 0.0

    # ATM approximation (avoids log(1)/0 issues near F≈K)
    if abs(log_FK) < atm_tolerance:
        # Hagan ATM formula
        A = alpha / (F ** (1.0 - beta))
        B = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2) / (F ** (2.0 - 2.0 * beta))
            + (rho * beta * nu * alpha) / (4.0 * F ** (1.0 - beta))
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
        ) * T
        return A * B

    # General formula
    z    = (nu / alpha) * FKm * log_FK
    chi  = np.log((np.sqrt(1.0 - 2.0 * rho * z + z ** 2) + z - rho) / (1.0 - rho))

    A = alpha / (
        FKm * (
            1.0
            + ((1.0 - beta) ** 2 / 24.0) * log_FK ** 2
            + ((1.0 - beta) ** 4 / 1920.0) * log_FK ** 4
        )
    )

    B = z / chi if abs(chi) > 1e-12 else 1.0

    C = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / FK ** (1.0 - beta)
        + (rho * beta * nu * alpha) / (4.0 * FK ** ((1.0 - beta) / 2.0))
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
    ) * T

    return A * B * C


def sabr_vol_vec(
    F: float,
    K_array: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Vectorised wrapper around sabr_vol for an array of strikes."""
    return np.array([sabr_vol(F, k, T, alpha, beta, rho, nu) for k in K_array])


# --------------------------------------------------------------------------- #
#  Calibration                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class SABRParams:
    alpha: float
    beta:  float
    rho:   float
    nu:    float
    rmse:  float = 0.0          # root mean squared error (in vol points)

    def __repr__(self):
        return (
            f"SABRParams(alpha={self.alpha:.6f}, beta={self.beta:.4f}, "
            f"rho={self.rho:.6f}, nu={self.nu:.6f}, rmse={self.rmse*100:.4f}%)"
        )


def calibrate_sabr(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
    weights: Optional[np.ndarray] = None,
    n_restarts: int = 5,
) -> SABRParams:
    """
    Calibrate SABR (alpha, rho, nu) for fixed beta using weighted least squares.

    Parameters
    ----------
    F            : forward swap rate
    T            : option expiry (years)
    strikes      : array of strikes
    market_vols  : array of market implied vols (same length as strikes)
    beta         : fixed CEV exponent (default 0.5)
    weights      : optional weight vector; defaults to uniform
    n_restarts   : number of random restarts to avoid local minima

    Returns
    -------
    SABRParams with calibrated (alpha, beta, rho, nu) and RMSE
    """
    if weights is None:
        weights = np.ones(len(strikes))
    weights = weights / weights.sum()

    # ATM vol → initial alpha guess
    atm_idx   = np.argmin(np.abs(strikes - F))
    sigma_atm = market_vols[atm_idx]
    alpha0    = sigma_atm * (F ** (1.0 - beta))

    bounds = [
        (1e-6, 5.0),    # alpha
        (-0.999, 0.999), # rho
        (1e-6, 5.0),    # nu
    ]

    def objective(params):
        a, r, v = params
        model_vols = sabr_vol_vec(F, strikes, T, a, beta, r, v)
        diff = model_vols - market_vols
        return np.sum(weights * diff ** 2)

    best_result = None
    rng = np.random.default_rng(42)

    for i in range(n_restarts):
        if i == 0:
            x0 = [alpha0, 0.0, 0.3]
        else:
            x0 = [
                rng.uniform(1e-4, 1.0),
                rng.uniform(-0.9, 0.9),
                rng.uniform(1e-4, 2.0),
            ]

        try:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        except Exception:
            pass

    if best_result is None or not best_result.success:
        warnings.warn("SABR calibration did not converge cleanly.")

    alpha, rho, nu = best_result.x
    model_vols = sabr_vol_vec(F, strikes, T, alpha, beta, rho, nu)
    rmse = np.sqrt(np.mean((model_vols - market_vols) ** 2))

    return SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu, rmse=rmse)


# --------------------------------------------------------------------------- #
#  Surface calibration                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class SABRSurface:
    """
    Holds calibrated SABR parameters for every (expiry, tenor) pair
    on the swaption surface.
    """
    results: dict = field(default_factory=dict)   # (expiry_str, tenor_str) → SABRParams
    forward_rates: dict = field(default_factory=dict)

    def add(self, expiry: str, tenor: str, params: SABRParams, F: float):
        self.results[(expiry, tenor)] = params
        self.forward_rates[(expiry, tenor)] = F

    def get(self, expiry: str, tenor: str) -> Optional[SABRParams]:
        return self.results.get((expiry, tenor))

    def implied_vol(
        self, expiry: str, tenor: str, K: float
    ) -> float:
        p = self.get(expiry, tenor)
        F = self.forward_rates.get((expiry, tenor))
        if p is None or F is None:
            raise KeyError(f"No calibration for ({expiry}, {tenor})")
        T = _parse_tenor(expiry)
        return sabr_vol(F, K, T, p.alpha, p.beta, p.rho, p.nu)

    def summary(self) -> pd.DataFrame:
        rows = []
        for (exp, ten), p in self.results.items():
            F = self.forward_rates[(exp, ten)]
            rows.append({
                "expiry": exp,
                "tenor":  ten,
                "F":      F,
                "alpha":  p.alpha,
                "beta":   p.beta,
                "rho":    p.rho,
                "nu":     p.nu,
                "rmse_%": p.rmse * 100,
            })
        df = pd.DataFrame(rows)
        # Sort by expiry then tenor numerically
        df["_exp_y"] = df["expiry"].str.replace("y", "").astype(float)
        df["_ten_y"] = df["tenor"].str.replace("y", "").astype(float)
        df = df.sort_values(["_exp_y", "_ten_y"]).drop(columns=["_exp_y", "_ten_y"])
        return df.reset_index(drop=True)


def _parse_tenor(tenor_str: str) -> float:
    tenor_str = tenor_str.strip().lower()
    if tenor_str.endswith("y"):
        return float(tenor_str[:-1])
    if tenor_str.endswith("m"):
        return float(tenor_str[:-1]) / 12.0
    raise ValueError(f"Cannot parse tenor: {tenor_str}")


def calibrate_surface(
    vol_surface_df: pd.DataFrame,
    yield_curve,
    beta: float = 0.5,
    n_restarts: int = 5,
) -> SABRSurface:
    """
    Calibrate SABR for every (expiry, tenor) pair present in vol_surface_df.

    Parameters
    ----------
    vol_surface_df : output of data_loader.load_full_vol_surface()
        columns: expiry, tenor, strike_spread_bps, atm_vol, implied_vol
    yield_curve    : YieldCurve instance
    beta           : fixed beta
    n_restarts     : restarts per slice

    Returns
    -------
    SABRSurface
    """
    surface = SABRSurface()

    pairs = vol_surface_df[["expiry", "tenor"]].drop_duplicates().values

    for expiry_str, tenor_str in pairs:
        slice_df = vol_surface_df[
            (vol_surface_df["expiry"] == expiry_str)
            & (vol_surface_df["tenor"]  == tenor_str)
        ].copy()

        T   = _parse_tenor(expiry_str)
        ten = _parse_tenor(tenor_str)
        F   = yield_curve.swap_rate(T, ten)

        # Convert strike spreads (bps) to absolute strikes
        slice_df["K"] = F + slice_df["strike_spread_bps"] / 10_000.0

        # Drop rows where vol is NaN or non-positive
        slice_df = slice_df.dropna(subset=["implied_vol"])
        slice_df = slice_df[slice_df["implied_vol"] > 0]

        if len(slice_df) < 3:
            warnings.warn(f"Too few data points for ({expiry_str}, {tenor_str}), skipping.")
            continue

        strikes     = slice_df["K"].values
        market_vols = slice_df["implied_vol"].values

        # Up-weight ATM (strike_spread==0)
        weights = np.where(slice_df["strike_spread_bps"].values == 0, 3.0, 1.0)

        params = calibrate_sabr(
            F, T, strikes, market_vols,
            beta=beta, weights=weights, n_restarts=n_restarts
        )

        surface.add(expiry_str, tenor_str, params, F)
        print(
            f"  ({expiry_str:>4s} x {tenor_str:>4s})  "
            f"F={F:.4%}  alpha={params.alpha:.5f}  "
            f"rho={params.rho:+.4f}  nu={params.nu:.5f}  "
            f"rmse={params.rmse*100:.4f}%"
        )

    return surface
