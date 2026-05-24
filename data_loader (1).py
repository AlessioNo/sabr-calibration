"""
data_loader.py
--------------
Lecture des données de marché :
  - Courbe OIS USD (usdois.xlsx)
  - Surface de vol implicite caps USD (cap_vol_surface__1_.xlsx)

La surface cap cotait les vols implicites log-normales en % sur une grille
120 expiries × 19 strikes (0.25% à 11%).

Convention de marché : les expiries 3M–1Y ont des vols identiques ;
le premier caplet liquide démarre à 1Y sur le marché USD 2016.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Discount factors OIS ──────────────────────────────────────────────────────

def load_discount_factors(path: str | Path) -> pd.DataFrame:
    """
    Retourne un DataFrame avec colonnes ['dates', 'discounts'].
    dates    : datetime
    discounts: float — P(0,T)
    """
    df = pd.read_excel(path)
    df['dates'] = pd.to_datetime(df['dates'])
    df = df.sort_values('dates').reset_index(drop=True)
    return df


# ── Surface de vol caps ────────────────────────────────────────────────────────

def load_cap_vol_surface(path: str | Path) -> pd.DataFrame:
    """
    Retourne la surface de vol caps au format tidy :
        expiry (str) | strike_pct (float) | implied_vol (float)

    implied_vol est en décimal (0.50 = 50%).
    strikes sont en décimal (0.02 = 2%).
    """
    raw = pd.read_excel(path, index_col=0)
    raw.index.name = 'expiry'
    raw.columns.name = 'strike_pct'

    # Convertir les colonnes strike en float (décimal)
    raw.columns = [float(c) / 100.0 for c in raw.columns]

    tidy = (
        raw.stack()
        .reset_index()
        .rename(columns={0: 'implied_vol'})
    )
    tidy['implied_vol'] = tidy['implied_vol'] / 100.0   # % → décimal
    tidy = tidy[tidy['implied_vol'] > 0].reset_index(drop=True)

    return tidy


# ── Smiles filtrées pour calibration ─────────────────────────────────────────

def load_calibration_smiles(
    path: str | Path,
    yield_curve,
    expiry_map: dict,
    K_min: float = 0.005,
    K_max: float = 0.06,
) -> list[dict]:
    """
    Prépare les smiles pour la calibration SABR.

    Pour chaque expiry dans expiry_map :
      - Forward bootstrappé depuis la courbe OIS
      - Strikes filtrés sur la zone liquide [K_min, K_max]
      - Vols de marché correspondantes

    Parameters
    ----------
    expiry_map : dict label → T en années  (ex: {'1Y': 1.0, '2Y': 2.0, ...})
    K_min, K_max : filtre de la zone liquide

    Returns
    -------
    list of dict : {label, expiry, F, strikes, mkt_vols}
    """
    raw = pd.read_excel(path, index_col=0)
    raw.columns = [float(c) / 100.0 for c in raw.columns]
    strikes_all = list(raw.columns)

    smiles = []
    for label, T in expiry_map.items():
        if label not in raw.index:
            continue
        F   = yield_curve.forward_rate(0, T)
        mkt = raw.loc[label].values / 100.0

        Ks = [K for K, v in zip(strikes_all, mkt) if K_min <= K <= K_max and v > 0.001]
        Vs = [v for K, v in zip(strikes_all, mkt) if K_min <= K <= K_max and v > 0.001]

        if len(Ks) < 3:
            continue

        smiles.append({
            'label'   : label,
            'expiry'  : T,
            'F'       : F,
            'strikes' : Ks,
            'mkt_vols': Vs,
        })

    return smiles


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    base = Path(__file__).parent

    df = load_discount_factors(base / 'usdois.xlsx')
    print('=== OIS ===')
    print(df.head())

    surf = load_cap_vol_surface(base / 'cap_vol_surface__1_.xlsx')
    print('\n=== Surface vol caps (sample) ===')
    print(surf[surf['expiry'] == '2Y'].head(10))
    print(f'\nDimension : {surf.shape[0]} observations')
