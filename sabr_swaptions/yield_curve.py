"""
yield_curve.py
--------------
Bootstraps a yield curve from market discount factors and provides:
  - Log-linear interpolation of P(0,T) for any T
  - Forward swap rate computation for (expiry, tenor) pairs
  - Annuity (PV01) computation

The reference date is inferred as the smallest Date in the discount factor file.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime
from scipy.interpolate import interp1d


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _to_years(d0: datetime, d1: datetime) -> float:
    """Act/365 day count."""
    return (d1 - d0).days / 365.0


def _parse_tenor(tenor_str: str) -> float:
    """Convert tenor string like '1y', '10y', '30y' to years (float)."""
    tenor_str = tenor_str.strip().lower()
    if tenor_str.endswith("y"):
        return float(tenor_str[:-1])
    if tenor_str.endswith("m"):
        return float(tenor_str[:-1]) / 12.0
    raise ValueError(f"Cannot parse tenor: {tenor_str}")


# --------------------------------------------------------------------------- #
#  YieldCurve class                                                             #
# --------------------------------------------------------------------------- #

class YieldCurve:
    """
    Log-linear interpolated yield curve built from (date, discount_factor) pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'Date' (datetime) and 'discount_factor' (float).
    """

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("Date").reset_index(drop=True)
        self.ref_date: datetime = df["Date"].iloc[0]

        # Store times-to-maturity (in years) and log discount factors
        self._T = np.array([_to_years(self.ref_date, d) for d in df["Date"]])
        self._P = df["discount_factor"].values.astype(float)
        self._logP = np.log(self._P)

        # Build log-linear interpolator (extrapolate flat beyond last pillar)
        self._interp = interp1d(
            self._T, self._logP,
            kind="linear",
            bounds_error=False,
            fill_value=(self._logP[0], self._logP[-1]),
        )

    # ------------------------------------------------------------------ #
    #  Core primitives                                                      #
    # ------------------------------------------------------------------ #

    def discount(self, T: float | np.ndarray) -> float | np.ndarray:
        """P(0, T) via log-linear interpolation."""
        return np.exp(self._interp(T))

    def zero_rate(self, T: float | np.ndarray) -> float | np.ndarray:
        """Continuously compounded zero rate r(T) = -ln P(0,T) / T."""
        T = np.asarray(T, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -np.log(self.discount(T)) / T
        return r

    # ------------------------------------------------------------------ #
    #  Swap instruments                                                     #
    # ------------------------------------------------------------------ #

    def annuity(
        self,
        T_start: float,
        tenor: float,
        freq: int = 1,
    ) -> float:
        """
        PV01 / annuity factor for a swap starting at T_start with given tenor.

        Parameters
        ----------
        T_start : float   – Option expiry in years
        tenor   : float   – Swap tenor in years
        freq    : int     – Coupon frequency per year (default 1 = annual)
        """
        dt = 1.0 / freq
        times = np.arange(T_start + dt, T_start + tenor + dt / 2, dt)
        return dt * np.sum(self.discount(times))

    def swap_rate(
        self,
        T_start: float,
        tenor: float,
        freq: int = 1,
    ) -> float:
        """
        Par (forward) swap rate S(T_start, tenor).

        S = (P(0, T_start) - P(0, T_start+tenor)) / annuity
        """
        A = self.annuity(T_start, tenor, freq)
        T_end = T_start + tenor
        return (self.discount(T_start) - self.discount(T_end)) / A

    # ------------------------------------------------------------------ #
    #  Convenience: batch computation from string labels                    #
    # ------------------------------------------------------------------ #

    def swap_rate_from_labels(
        self, expiry_str: str, tenor_str: str, freq: int = 1
    ) -> float:
        """e.g. swap_rate_from_labels('5y', '10y')"""
        T_start = _parse_tenor(expiry_str)
        tenor   = _parse_tenor(tenor_str)
        return self.swap_rate(T_start, tenor, freq)

    def annuity_from_labels(
        self, expiry_str: str, tenor_str: str, freq: int = 1
    ) -> float:
        T_start = _parse_tenor(expiry_str)
        tenor   = _parse_tenor(tenor_str)
        return self.annuity(T_start, tenor, freq)

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def pillar_summary(self) -> pd.DataFrame:
        """Returns a table of all pillars with T, P(0,T), and zero rate."""
        return pd.DataFrame({
            "T_years": self._T,
            "P(0,T)": self._P,
            "zero_rate": self.zero_rate(self._T),
        })


# --------------------------------------------------------------------------- #
#  Quick sanity check                                                           #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from pathlib import Path
    from data_loader import load_discount_factors

    path = Path(__file__).parent / "data" / "market_discount_factors.xlsx"
    df   = load_discount_factors(path)
    yc   = YieldCurve(df)

    print("=== Yield Curve Pillars ===")
    print(yc.pillar_summary().to_string(index=False))

    print("\n=== Sample Swap Rates ===")
    for exp, ten in [("1y", "5y"), ("2y", "10y"), ("5y", "10y"), ("10y", "10y")]:
        s = yc.swap_rate_from_labels(exp, ten)
        print(f"  S({exp}, {ten}) = {s:.4%}")
