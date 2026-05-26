"""
data_loader.py
--------------
Loads market discount factors and swaption volatility quotes from Excel files.

Swaption vol strikes sheet encodes vols as DIFFERENCES from ATM vol:
    abs_vol(K) = atm_vol(expiry, tenor) + spread(expiry, tenor, K)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Discount factors                                                             #
# --------------------------------------------------------------------------- #

def load_discount_factors(path: str | Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['Date', 'P(0,T)'].
    Date is parsed as datetime, P(0,T) is float.
    """
    df = pd.read_excel(path, sheet_name="Sheet1", parse_dates=["Date"])
    df = df.rename(columns={"P(0,T)": "discount_factor"})
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
#  Swaption ATM vols                                                            #
# --------------------------------------------------------------------------- #

def load_atm_vols(path: str | Path) -> pd.DataFrame:
    """
    Returns a tidy DataFrame:
        expiry (str)  | tenor (str) | atm_vol (float)

    Original sheet has expiries as rows and tenors as columns.
    """
    raw = pd.read_excel(path, sheet_name="swaption ATM vol", index_col=0)
    raw.index.name = "expiry"
    raw.columns.name = "tenor"

    tidy = (
        raw.stack()
        .reset_index()
        .rename(columns={0: "atm_vol"})
    )
    return tidy


# --------------------------------------------------------------------------- #
#  Swaption smile (vol spreads by strike)                                       #
# --------------------------------------------------------------------------- #

def load_vol_spreads(path: str | Path) -> pd.DataFrame:
    """
    Returns a tidy DataFrame:
        expiry (str) | tenor (str) | strike_spread_bps (int) | vol_spread (float)

    Strike column headers are in basis points relative to ATM (e.g. -200 = ATM-200bps).
    Vol values are DIFFERENCES from ATM vol (can be negative).
    """
    raw = pd.read_excel(path, sheet_name="swaption vol strikes")

    # First two columns are expiry and tenor; the rest are strike spreads
    id_cols = raw.columns[:2].tolist()          # ['Expiry Tenor\\Strike', col2] → rename
    strike_cols = raw.columns[2:].tolist()       # [-200, -100, -50, -25, 25, 50, 100, 200]

    raw = raw.rename(columns={raw.columns[0]: "expiry", raw.columns[1]: "tenor"})
    raw["expiry"] = raw["expiry"].astype(str).str.strip()
    raw["tenor"]  = raw["tenor"].astype(str).str.strip()

    tidy = raw.melt(
        id_vars=["expiry", "tenor"],
        value_vars=strike_cols,
        var_name="strike_spread_bps",
        value_name="vol_spread",
    )
    tidy["strike_spread_bps"] = tidy["strike_spread_bps"].astype(int)
    return tidy


# --------------------------------------------------------------------------- #
#  Combined surface: absolute vols at each strike                               #
# --------------------------------------------------------------------------- #

def load_full_vol_surface(path: str | Path) -> pd.DataFrame:
    """
    Merges ATM vols and spreads to produce absolute implied vols at every strike.

    Returns a tidy DataFrame:
        expiry | tenor | strike_spread_bps | atm_vol | vol_spread | implied_vol
    """
    atm   = load_atm_vols(path)
    spreads = load_vol_spreads(path)

    # ATM entry has zero spread
    atm_entry = atm.copy()
    atm_entry["strike_spread_bps"] = 0
    atm_entry["vol_spread"] = 0.0

    # Merge spreads with ATM to get absolute vols
    merged = spreads.merge(atm, on=["expiry", "tenor"], how="left")
    merged["implied_vol"] = merged["atm_vol"] + merged["vol_spread"]

    # Append ATM strikes
    atm_entry["implied_vol"] = atm_entry["atm_vol"]
    full = pd.concat([merged, atm_entry], ignore_index=True)
    full = full.sort_values(["expiry", "tenor", "strike_spread_bps"]).reset_index(drop=True)

    return full


# --------------------------------------------------------------------------- #
#  Quick sanity check                                                           #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    base = Path(__file__).parent / "data"
    df_path  = base / "market_discount_factors.xlsx"
    sw_path  = base / "swaption_quotes.xlsx"

    print("=== Discount Factors ===")
    df = load_discount_factors(df_path)
    print(df.head())

    print("\n=== ATM Vols ===")
    atm = load_atm_vols(sw_path)
    print(atm.head(15))

    print("\n=== Vol Spreads ===")
    sp = load_vol_spreads(sw_path)
    print(sp.head(15))

    print("\n=== Full Surface (sample) ===")
    surf = load_full_vol_surface(sw_path)
    print(surf[surf.expiry == "5y"].head(20))
