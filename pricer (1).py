"""
pricer.py
---------
Pricing et Greeks pour caps et floors sous le modèle de Black (1976).

La volatilité utilisée est extraite de la surface SABR calibrée,
ce qui capture l'effet smile sur le prix.

Parité Cap–Floor :
    Cap(K) − Floor(K) = PV(Swap payeur K)

Les floors sont déduits par parité depuis les caps et la courbe OIS.

API publique
------------
  black_caplet(F, K, sigma, T, notional, delta, discount, type_)
      → float  (prix en €)

  cap_price(yield_curve, surface, K, T_mat, notional, delta, shift)
      → (float, pd.DataFrame)  (prix total + décomposition caplet par caplet)

  floor_price(yield_curve, surface, K, T_mat, notional, delta, shift)
      → float  (prix par parité)

  cap_greeks(yield_curve, surface, K, T_mat, notional, delta, shift)
      → dict  {delta_bp, vega_1pct, gamma_bp2, vanna, volga}
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Black (1976) — caplet / floorlet ─────────────────────────────────────────

def black_caplet(
    F: float,
    K: float,
    sigma: float,
    T: float,
    notional: float,
    delta: float,
    discount: float,
    type_: str = 'cap',
) -> float:
    """
    Prix d'un caplet ou floorlet — formule de Black (1976).

    Parameters
    ----------
    F        : taux forward LIBOR
    K        : strike
    sigma    : vol implicite Black (log-normale) — extraite du SABR calibré
    T        : expiry du caplet (années)
    notional : notionnel (€)
    delta    : longueur de la période (ex: 0.5 pour 6M)
    discount : facteur d'actualisation P(0, T_end)
    type_    : 'cap' (call sur taux) ou 'floor' (put sur taux)
    """
    if sigma <= 0 or T <= 0 or F <= 0 or K <= 0:
        # Valeur intrinsèque
        if type_ == 'cap':
            return notional * delta * discount * max(F - K, 0.0)
        return notional * delta * discount * max(K - F, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if type_ == 'cap':
        return notional * delta * discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return notional * delta * discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ── Cap complet (somme de caplets) ────────────────────────────────────────────

def cap_price(
    yield_curve,
    surface,
    K: float,
    T_mat: float,
    notional: float = 1_000_000,
    delta: float = 0.5,
    shift: float = 0.01,
    type_: str = 'cap',
) -> tuple[float, pd.DataFrame]:
    """
    Price un cap (ou floor) de maturité T_mat et strike K.

    Un cap est la somme des caplets : Σ Caplet(Fᵢ, K, Tᵢ)
    Chaque caplet utilise :
      - le forward bootstrappé depuis la courbe OIS (données réelles)
      - la vol SABR calibrée interpolée pour son expiry

    Parameters
    ----------
    yield_curve : YieldCurve
    surface     : SABRCapSurface avec interpolateur de paramètres
    K           : strike (décimal, ex: 0.02 = 2%)
    T_mat       : maturité du cap (années)
    notional    : notionnel (€)
    delta       : fréquence (0.5 = semi-annuel)
    shift       : décalage Shifted SABR
    type_       : 'cap' ou 'floor'

    Returns
    -------
    (prix_total, df_caplets) — prix en € et décomposition par caplet
    """
    total   = 0.0
    rows    = []
    T_s     = 0.0
    T_e     = delta

    while T_e <= T_mat + 1e-8:
        F   = yield_curve.forward_rate(T_s, T_e)
        P   = float(yield_curve.discount(T_e))
        sig = surface.vol_at(T_e, K, shift)
        px  = black_caplet(F, K, sig, T_e, notional, delta, P, type_)

        total += px
        rows.append({
            'T_start'            : round(T_s, 2),
            'T_end'              : round(T_e, 2),
            f'F (%)'             : round(F * 100, 4),
            f'sigma_SABR (%)'    : round(sig * 100, 4),
            f'Prix {type_} ($)'  : round(px, 4),
        })
        T_s = T_e
        T_e = round(T_s + delta, 4)

    df = pd.DataFrame(rows)
    df['Cumule ($)'] = df[f'Prix {type_} ($)'].cumsum().round(4)
    return total, df


def floor_price(
    yield_curve,
    surface,
    K: float,
    T_mat: float,
    notional: float = 1_000_000,
    delta: float = 0.5,
    shift: float = 0.01,
) -> float:
    """
    Prix d'un floor par parité Cap–Floor–Swap :
        Floor(K) = Cap(K) − PV(Swap payeur K)

    Aucune vol supplémentaire nécessaire — uniquement la courbe OIS.
    """
    px_cap, _ = cap_price(yield_curve, surface, K, T_mat, notional, delta, shift, 'cap')
    pv_sw     = yield_curve.pv_swap(K, 0.0, T_mat, delta, notional)
    return px_cap - pv_sw


# ── Greeks analytiques (Black) ────────────────────────────────────────────────

def cap_greeks(
    yield_curve,
    surface,
    K: float,
    T_mat: float,
    notional: float = 1_000_000,
    delta: float = 0.5,
    shift: float = 0.01,
) -> dict:
    """
    Greeks agrégés du cap — somme des Greeks de chaque caplet.

    Retourne
    --------
    dict : delta_bp, vega_1pct, gamma_bp2, vanna, volga
        delta_bp  : ∂Prix/∂F en €/bp
        vega_1pct : ∂Prix/∂σ en € pour +1% de vol
        gamma_bp2 : ∂²Prix/∂F² en €/bp²
        vanna     : ∂²Prix/∂F∂σ en €/(bp × 1%)
        volga     : ∂²Prix/∂σ² en €/(1%)²
    """
    greeks = {'delta_bp': 0.0, 'vega_1pct': 0.0,
              'gamma_bp2': 0.0, 'vanna': 0.0, 'volga': 0.0}

    T_s = 0.0
    T_e = delta
    while T_e <= T_mat + 1e-8:
        F   = yield_curve.forward_rate(T_s, T_e)
        P   = float(yield_curve.discount(T_e))
        sig = surface.vol_at(T_e, K, shift)

        if sig > 0 and T_e > 0 and F > 0 and K > 0:
            sqrtT = math.sqrt(T_e)
            d1 = (math.log(F / K) + 0.5*sig**2*T_e) / (sig*sqrtT)
            d2 = d1 - sig*sqrtT
            phi = norm.pdf(d1)
            w   = notional * delta * P

            greeks['delta_bp']  += w * norm.cdf(d1)  * 1e-4
            greeks['vega_1pct'] += w * F * phi * sqrtT * 0.01
            greeks['gamma_bp2'] += w * phi / (F * sig * sqrtT) * (1e-4)**2
            greeks['vanna']     += w * (-phi * d2 / sig) * 1e-4 * 0.01
            greeks['volga']     += w * F * phi * sqrtT * d1 * d2 / sig * (0.01)**2

        T_s = T_e
        T_e = round(T_s + delta, 4)

    return greeks
