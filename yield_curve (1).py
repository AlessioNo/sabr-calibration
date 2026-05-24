"""
yield_curve.py
--------------
Bootstrap de la courbe OIS depuis les discount factors de marché.

Fournit :
  - P(0,T)            : interpolation log-linéaire
  - forward_rate()    : taux LIBOR 6M forward bootstrappé (relation d'arbitrage exacte)
  - annuity()         : PV01 pour le pricing des floors (parité)
  - zero_rate()       : taux zéro-coupon continu

Convention : Act/365. Interpolation log-linéaire standard marché.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d


def _to_years(d0: datetime, d1: datetime) -> float:
    """Act/365 day count."""
    return (d1 - d0).days / 365.0


class YieldCurve:
    """
    Courbe OIS construite par interpolation log-linéaire des P(0,T).

    Parameters
    ----------
    df : pd.DataFrame
        Colonnes 'dates' (datetime) et 'discounts' (float).
    """

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df['dates'] = pd.to_datetime(df['dates'])
        df = df.sort_values('dates').reset_index(drop=True)

        self.ref_date = df['dates'].iloc[0]
        self._T = np.array([_to_years(self.ref_date, d) for d in df['dates']])
        self._P = df['discounts'].values.astype(float)
        self._logP = np.log(self._P)

        # Interpolateur log-linéaire — extrapolation plate hors des piliers
        self._interp = interp1d(
            self._T, self._logP,
            kind='linear',
            bounds_error=False,
            fill_value=(self._logP[0], self._logP[-1]),
        )

    # ── Primitives ────────────────────────────────────────────────────────────

    def discount(self, T: float | np.ndarray) -> float | np.ndarray:
        """P(0,T) par interpolation log-linéaire."""
        return np.exp(self._interp(np.maximum(T, 1e-9)))

    def zero_rate(self, T: float | np.ndarray) -> float | np.ndarray:
        """Taux zéro-coupon continu r(T) = -ln P(0,T) / T."""
        T = np.asarray(T, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return -np.log(self.discount(T)) / T

    # ── Taux forward LIBOR 6M bootstrappé ─────────────────────────────────────

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Taux forward LIBOR entre T1 et T2, bootstrappé depuis l'OIS.

        Relation d'arbitrage exacte :
            F(T1, T2) = (P(0,T1) / P(0,T2) - 1) / (T2 - T1)

        Note : ce n'est PAS une donnée synthétique — c'est une conséquence
        directe des vrais prix OIS observés.
        """
        return (self.discount(T1) / self.discount(T2) - 1) / (T2 - T1)

    def forward_curve(self, delta: float = 0.5, T_max: float = 30.0) -> pd.DataFrame:
        """
        Courbe complète de taux forwards LIBOR delta-périodiques.

        Returns
        -------
        DataFrame : T_start, T_end, F, P(0,T2), delta
        """
        rows = []
        T_s = 0.0
        T_e = delta
        while T_e <= T_max + 1e-8:
            rows.append({
                'T_start'  : round(T_s, 4),
                'T_end'    : round(T_e, 4),
                'F'        : self.forward_rate(T_s, T_e),
                'P(0,T2)'  : self.discount(T_e),
                'delta'    : delta,
            })
            T_s = T_e
            T_e = round(T_s + delta, 4)
        return pd.DataFrame(rows)

    # ── PV01 / Annuité (pour pricing floors par parité) ────────────────────────

    def annuity(self, T_start: float, T_end: float, delta: float = 0.5) -> float:
        """
        Annuité d'un cap/floor entre T_start et T_end.
        A = Σ δ × P(0, Tᵢ)  pour Tᵢ ∈ ]T_start, T_end]
        """
        times = np.arange(T_start + delta, T_end + delta / 2, delta)
        return delta * float(np.sum(self.discount(times)))

    def pv_swap(self, K: float, T_start: float, T_end: float,
                delta: float = 0.5, notional: float = 1e6) -> float:
        """
        PV d'un swap payeur fixe K, de T_start à T_end.
        PV = notional × Σ δ × P(0,Tᵢ) × (Fᵢ - K)
        Utilisé pour déduire les floors depuis les caps (parité).
        """
        pv = 0.0
        T_s = T_start
        T_e = round(T_s + delta, 4)
        while T_e <= T_end + 1e-8:
            F = self.forward_rate(T_s, T_e)
            pv += delta * self.discount(T_e) * (F - K)
            T_s = T_e
            T_e = round(T_s + delta, 4)
        return notional * pv

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def pillar_summary(self) -> pd.DataFrame:
        """Tableau des piliers : T, P(0,T), taux zéro."""
        return pd.DataFrame({
            'T (ans)'      : np.round(self._T, 4),
            'P(0,T)'       : np.round(self._P, 6),
            'Taux zero (%)': np.round(self.zero_rate(self._T) * 100, 4),
        })
