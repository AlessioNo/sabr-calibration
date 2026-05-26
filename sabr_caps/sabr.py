"""
sabr.py
-------
Modèle SABR (Hagan et al. 2002) pour caps et floors.

Fournit :
  sabr_vol()           : vol implicite log-normale (scalaire)
  sabr_vol_vec()       : version vectorisée sur un tableau de strikes
  shifted_sabr_vol()   : Shifted SABR (shift = 1% par défaut)
  calibrate_sabr()     : calibration (alpha, rho, nu) pour beta fixé
  SABRCapSurface       : surface complète calibrée
  calibrate_surface()  : calibre toutes les slices
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ── Formule Hagan 2002 ────────────────────────────────────────────────────────

def sabr_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    atm_tol: float = 1e-7,
) -> float:
    """
    Vol implicite log-normale SABR — expansion asymptotique Hagan 2002.

    Parameters
    ----------
    F     : taux forward (LIBOR 6M bootstrappé)
    K     : strike
    T     : expiry du caplet (années)
    alpha : vol initiale (> 0)
    beta  : exposant CEV ∈ [0,1]
    rho   : corrélation F–alpha ∈ (−1,1)
    nu    : vol of vol (> 0)

    Returns
    -------
    Vol implicite log-normale (Black)
    """
    if T <= 0 or K <= 0 or F <= 0:
        return 1e-6

    FK  = F * K
    FKm = FK ** ((1.0 - beta) / 2.0)
    log_FK = np.log(F / K) if abs(F / K - 1.0) > atm_tol else 0.0

    # ── Cas ATM (F ≈ K) ───────────────────────────────────────────────────────
    if abs(log_FK) < atm_tol:
        A = alpha / (F ** (1.0 - beta))
        B = 1.0 + (
            (1.0 - beta)**2 / 24.0 * alpha**2 / F**(2.0 - 2.0*beta)
            + rho * beta * nu * alpha / (4.0 * F**(1.0 - beta))
            + (2.0 - 3.0*rho**2) / 24.0 * nu**2
        ) * T
        return max(A * B, 1e-6)

    # ── Cas non-ATM ───────────────────────────────────────────────────────────
    z   = (nu / alpha) * FKm * log_FK
    chi = np.log((np.sqrt(1.0 - 2.0*rho*z + z**2) + z - rho) / (1.0 - rho))

    A = alpha / (
        FKm * (
            1.0
            + (1.0 - beta)**2 / 24.0 * log_FK**2
            + (1.0 - beta)**4 / 1920.0 * log_FK**4
        )
    )
    B = z / chi if abs(chi) > 1e-12 else 1.0
    C = 1.0 + (
        (1.0 - beta)**2 / 24.0 * alpha**2 / FK**(1.0 - beta)
        + rho * beta * nu * alpha / (4.0 * FK**((1.0 - beta) / 2.0))
        + (2.0 - 3.0*rho**2) / 24.0 * nu**2
    ) * T

    return max(A * B * C, 1e-6)


def sabr_vol_vec(
    F: float,
    K_array: np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Version vectorisée de sabr_vol sur un tableau de strikes."""
    return np.array([sabr_vol(F, k, T, alpha, beta, rho, nu) for k in K_array])


def shifted_sabr_vol(
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    F: float,
    K: float,
    T: float,
    shift: float = 0.01,
) -> float:
    """
    Shifted SABR : décale F et K de 'shift' pour stabiliser
    la calibration en environnement de taux bas.
    Shift = 1% (100 bps) standard marché USD/EUR 2016.
    """
    return sabr_vol(F + shift, K + shift, T, alpha, beta, rho, nu)


# ── Résultats de calibration ──────────────────────────────────────────────────

@dataclass
class SABRParams:
    alpha : float
    beta  : float
    rho   : float
    nu    : float
    rmse  : float = 0.0     # RMSE en points de vol (décimal)

    def __repr__(self):
        return (
            f'SABRParams(alpha={self.alpha:.6f}, beta={self.beta:.2f}, '
            f'rho={self.rho:.6f}, nu={self.nu:.6f}, '
            f'rmse={self.rmse*10000:.2f} bps)'
        )


# ── Calibration d'une smile ───────────────────────────────────────────────────

def calibrate_sabr(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 0.5,
    shift: float = 0.01,
    n_restarts: int = 5,
) -> SABRParams:
    """
    Calibre (alpha, rho, nu) pour beta fixé sur une smile de caps.

    Méthode : moindres carrés pondérés (L-BFGS-B), n_restarts points de départ.
    Pondération x3 sur le strike ATM (point le plus liquide).
    Initialisation : alpha0 = sigma_ATM × (F+shift)^(1-beta)

    Parameters
    ----------
    F, T         : forward et expiry du caplet
    strikes      : tableau de strikes
    market_vols  : vols implicites de marché correspondantes
    beta         : exposant CEV fixé
    shift        : décalage Shifted SABR
    n_restarts   : restarts aléatoires

    Returns
    -------
    SABRParams calibré
    """
    strikes     = np.asarray(strikes)
    market_vols = np.asarray(market_vols)

    # Pondération : x3 sur le strike ATM
    weights = np.where(np.abs(strikes - F) == np.min(np.abs(strikes - F)), 3.0, 1.0)
    weights = weights / weights.sum()

    # Initialisation alpha depuis vol ATM observée
    idx_atm = np.argmin(np.abs(strikes - F))
    alpha0  = max(market_vols[idx_atm] * ((F + shift) ** (1.0 - beta)), 1e-5)

    bounds = [(1e-5, 50.0), (-0.999, 0.999), (1e-5, 10.0)]

    def objective(params):
        a, r, n = params
        model = sabr_vol_vec(F + shift, strikes + shift, T, a, beta, r, n)
        diff  = model - market_vols
        return float(np.sum(weights * diff**2))

    best_result = None
    rng = np.random.default_rng(42)

    for i in range(n_restarts):
        if i == 0:
            x0 = [alpha0, -0.20, 0.50]
        else:
            x0 = [
                rng.uniform(1e-4, min(alpha0 * 3, 5.0)),
                rng.uniform(-0.95, 0.0),
                rng.uniform(1e-4, 2.0),
            ]
        try:
            res = minimize(
                objective, x0, method='L-BFGS-B', bounds=bounds,
                options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 3000},
            )
            if best_result is None or res.fun < best_result.fun:
                best_result = res
        except Exception:
            pass

    if best_result is None:
        warnings.warn(f'Calibration failed for F={F:.4%}, T={T:.1f}Y')
        return SABRParams(alpha=alpha0, beta=beta, rho=0.0, nu=0.3, rmse=1.0)

    alpha_c, rho_c, nu_c = best_result.x
    model_vols = sabr_vol_vec(F + shift, strikes + shift, T, alpha_c, beta, rho_c, nu_c)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols)**2)))

    return SABRParams(alpha=alpha_c, beta=beta, rho=rho_c, nu=nu_c, rmse=rmse)


# ── Surface de paramètres calibrés ────────────────────────────────────────────

@dataclass
class SABRCapSurface:
    """
    Stocke les paramètres SABR calibrés pour chaque expiry de la surface caps.
    """
    results     : dict = field(default_factory=dict)   # label → SABRParams
    forward_rates: dict = field(default_factory=dict)  # label → F
    expiries    : dict = field(default_factory=dict)   # label → T (années)

    def add(self, label: str, T: float, F: float, params: SABRParams):
        self.results[label]      = params
        self.forward_rates[label] = F
        self.expiries[label]     = T

    def get(self, label: str) -> Optional[SABRParams]:
        return self.results.get(label)

    def implied_vol(self, label: str, K: float, shift: float = 0.01) -> float:
        """Vol SABR implicite pour un strike K donné."""
        p = self.get(label)
        F = self.forward_rates.get(label)
        T = self.expiries.get(label)
        if p is None:
            raise KeyError(f'Pas de calibration pour {label}')
        return sabr_vol(F + shift, K + shift, T, p.alpha, p.beta, p.rho, p.nu)

    def build_interpolators(self):
        """
        Construit des interpolateurs linéaires sur T pour (alpha, rho, nu).
        Permet de pricer des caps de maturité quelconque (pas seulement les expiries calibrées).
        """
        from scipy.interpolate import interp1d
        T_cal = np.array(sorted(self.expiries.values()))
        lbls  = sorted(self.expiries.keys(), key=lambda x: self.expiries[x])
        self._i_alpha = interp1d(T_cal, [self.results[l].alpha for l in lbls],
                                  kind='linear', fill_value='extrapolate')
        self._i_rho   = interp1d(T_cal, [self.results[l].rho   for l in lbls],
                                  kind='linear', fill_value='extrapolate')
        self._i_nu    = interp1d(T_cal, [self.results[l].nu    for l in lbls],
                                  kind='linear', fill_value='extrapolate')
        self._beta    = list(self.results.values())[0].beta

    def vol_at(self, T: float, K: float, shift: float = 0.01) -> float:
        """
        Vol SABR pour une expiry T quelconque (interpolée) et un strike K.
        Utilisé par le pricer pour les caps multi-caplets.
        """
        if not hasattr(self, '_i_alpha'):
            self.build_interpolators()
        # Forward bootstrappé depuis la courbe OIS (stocké dans forward_rates)
        # On approche F depuis les calibrations disponibles
        T_vals = sorted(self.expiries.values())
        F_vals = [self.forward_rates[l] for l in sorted(self.expiries.keys(),
                   key=lambda x: self.expiries[x])]
        from scipy.interpolate import interp1d as i1d
        F = float(i1d(T_vals, F_vals, kind='linear', fill_value='extrapolate'
                      )(T))
        alpha = float(self._i_alpha(T))
        rho   = float(self._i_rho(T))
        nu    = float(self._i_nu(T))
        return sabr_vol(F + shift, K + shift, T, alpha, self._beta, rho, nu)

    def summary(self) -> pd.DataFrame:
        """DataFrame des paramètres calibrés triés par expiry."""
        rows = []
        for label, p in self.results.items():
            rows.append({
                'expiry'  : label,
                'T (ans)' : self.expiries[label],
                'F (%)'   : round(self.forward_rates[label] * 100, 4),
                'alpha'   : p.alpha,
                'beta'    : p.beta,
                'rho'     : p.rho,
                'nu'      : p.nu,
                'rmse_%'  : p.rmse * 100,
            })
        df = pd.DataFrame(rows)
        df = df.sort_values('T (ans)').reset_index(drop=True)
        return df


# ── Pipeline de calibration complète ─────────────────────────────────────────

def calibrate_surface(
    smiles: list[dict],
    beta: float = 0.0,
    shift: float = 0.01,
    n_restarts: int = 5,
) -> SABRCapSurface:
    """
    Calibre SABR pour chaque smile de la liste.

    Parameters
    ----------
    smiles     : sortie de data_loader.load_calibration_smiles()
    beta       : exposant CEV fixé
    shift      : décalage Shifted SABR
    n_restarts : restarts par slice

    Returns
    -------
    SABRCapSurface
    """
    surface = SABRCapSurface()

    for s in smiles:
        params = calibrate_sabr(
            F=s['F'], T=s['expiry'],
            strikes=np.array(s['strikes']),
            market_vols=np.array(s['mkt_vols']),
            beta=beta, shift=shift, n_restarts=n_restarts,
        )
        surface.add(s['label'], s['expiry'], s['F'], params)
        print(
            f"  ({s['label']:>4s})  F={s['F']:.4%}  "
            f"alpha={params.alpha:.5f}  rho={params.rho:+.4f}  "
            f"nu={params.nu:.5f}  rmse={params.rmse*10000:.2f} bps"
        )

    return surface


# ── Calibration avec beta libre (4 paramètres) ───────────────────────────────

def calibrate_sabr_beta_free(
    F: float,
    T: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    shift: float = 0.01,
    n_restarts: int = 5,
) -> SABRParams:
    """
    Calibre (alpha, beta, rho, nu) — beta est un 4ème paramètre libre ∈ [0,1].

    Utile quand beta fixé à 0.5 donne un mauvais fit.
    Sur les caps USD 2016, beta converge vers 0–0.3 (régime quasi-normal),
    ce qui réduit le RMSE de ~88% vs beta=0.5 fixé.
    """
    strikes     = np.asarray(strikes)
    market_vols = np.asarray(market_vols)

    weights = np.where(np.abs(strikes - F) == np.min(np.abs(strikes - F)), 3.0, 1.0)
    weights = weights / weights.sum()

    def objective(params):
        a, b, r, n = params
        model = sabr_vol_vec(F + shift, strikes + shift, T, a, b, r, n)
        diff  = model - market_vols
        return float(np.sum(weights * diff**2))

    best_result = None
    rng = np.random.default_rng(42)

    for b0 in [0.0, 0.2, 0.5, 0.7]:
        idx    = np.argmin(np.abs(strikes - F))
        alpha0 = max(market_vols[idx] * ((F + shift) ** (1.0 - b0)), 1e-5)
        for i in range(n_restarts):
            if i == 0:
                x0 = [alpha0, b0, -0.20, 0.50]
            else:
                x0 = [
                    rng.uniform(1e-4, min(alpha0 * 3, 5.0)),
                    rng.uniform(0.0, 1.0),
                    rng.uniform(-0.95, 0.0),
                    rng.uniform(1e-4, 2.0),
                ]
            try:
                res = minimize(
                    objective, x0, method='L-BFGS-B',
                    bounds=[(1e-5,50),(0,1),(-0.999,0.999),(1e-5,10)],
                    options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 3000},
                )
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
            except Exception:
                pass

    alpha_c, beta_c, rho_c, nu_c = best_result.x
    model_vols = sabr_vol_vec(F + shift, strikes + shift, T, alpha_c, beta_c, rho_c, nu_c)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

    return SABRParams(alpha=alpha_c, beta=beta_c, rho=rho_c, nu=nu_c, rmse=rmse)
