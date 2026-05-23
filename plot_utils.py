"""
plot_utils.py
-------------
Matplotlib visualisation helpers for the SABR calibration notebook.

All heavy plotting code lives here so the notebook cells reduce to a
single function call each.

Public API
----------
  sort_index_numeric(df)
  plot_sabr_params_tables(summary)
  plot_sabr_heatmaps(summary)
  plot_all_smiles(surface, vol_surface)
  plot_rmse(summary, beta)
  plot_individual_smile(surface, vol_surface, expiry, tenor)
  plot_price_vega_smile(surface, vol_surface, yc,
                        expiry, tenor, strike_bps,
                        notional, option_type)
  plot_beta_comparison(vol_surface, yc, expiry, tenor, betas, n_restarts)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sabr        import sabr_vol_vec, calibrate_sabr, _parse_tenor
from pricer      import swaption_price, black_greeks


# --------------------------------------------------------------------------- #
#  Utility                                                                      #
# --------------------------------------------------------------------------- #

def sort_index_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort both rows and columns of a pivot DataFrame in ascending numeric order.

    Labels are expected to end in 'y' or 'm'  (e.g. '5y', '10y', '6m').
    """
    def _key(label: str) -> float:
        label = label.strip().lower()
        if label.endswith("y"):
            return float(label[:-1])
        if label.endswith("m"):
            return float(label[:-1]) / 12.0
        return float(label)

    row_order = sorted(df.index,   key=_key)
    col_order = sorted(df.columns, key=_key)
    return df.loc[row_order, col_order]


# --------------------------------------------------------------------------- #
#  Section 4 — parameter tables                                                 #
# --------------------------------------------------------------------------- #

def plot_sabr_params_tables(summary: pd.DataFrame) -> None:
    """
    Print pivot tables for α, ρ, ν and RMSE (%) sorted numerically.
    """
    for col, label in [
        ("alpha", "α (alpha)"),
        ("rho",   "ρ (rho)"),
        ("nu",    "ν (nu)"),
        ("rmse_%","RMSE (%)"),
    ]:
        pivot = summary.pivot(index="expiry", columns="tenor", values=col)
        pivot = sort_index_numeric(pivot)
        print(f"\n── {label} ──")
        print(pivot.to_string())


# --------------------------------------------------------------------------- #
#  Section 5 — parameter heatmaps                                               #
# --------------------------------------------------------------------------- #

def plot_sabr_heatmaps(summary: pd.DataFrame) -> None:
    """
    3-panel heatmap: α, ρ, ν across the (expiry × tenor) surface.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col, label in zip(
        axes,
        ["alpha", "rho", "nu"],
        ["α (alpha)", "ρ (rho)", "ν (nu)"],
    ):
        pivot = sort_index_numeric(
            summary.pivot(index="expiry", columns="tenor", values=col)
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(label, fontsize=13)
        plt.colorbar(im, ax=ax)

    fig.suptitle("SABR Parameter Heatmaps", fontsize=14)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  Section 6 — all smiles (market vs. SABR)                                    #
# --------------------------------------------------------------------------- #

def plot_all_smiles(surface, vol_surface: pd.DataFrame) -> None:
    """
    Grid of subplots — one per (expiry, tenor) slice — overlaying the
    calibrated SABR smile against market implied vols.
    """
    pairs = list(surface.results.keys())
    ncols = 4
    nrows = -(-len(pairs) // ncols)   # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes_flat = axes.flatten()

    for ax, (expiry_str, tenor_str) in zip(axes_flat, pairs):
        T   = _parse_tenor(expiry_str)
        F   = surface.forward_rates[(expiry_str, tenor_str)]
        p   = surface.results[(expiry_str, tenor_str)]

        mkt = vol_surface[
            (vol_surface["expiry"] == expiry_str) &
            (vol_surface["tenor"]  == tenor_str)
        ].copy()
        mkt["K"] = F + mkt["strike_spread_bps"] / 10_000.0

        K_grid     = np.linspace(mkt["K"].min() * 0.98, mkt["K"].max() * 1.02, 200)
        model_vols = sabr_vol_vec(F, K_grid, T, p.alpha, p.beta, p.rho, p.nu)

        ax.plot(K_grid * 100, model_vols * 100, "b-", lw=1.5, label="SABR")
        ax.scatter(
            mkt["K"] * 100, mkt["implied_vol"] * 100,
            color="red", zorder=5, s=35, label="Marché",
        )
        ax.axvline(F * 100, ls="--", color="grey", lw=0.8)
        ax.set_title(
            f"{expiry_str} × {tenor_str}\n"
            f"α={p.alpha:.4f}  ρ={p.rho:+.3f}  ν={p.nu:.4f}",
            fontsize=9,
        )
        ax.set_xlabel("Strike (%)", fontsize=8)
        ax.set_ylabel("Vol (%)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle("SABR Smile Calibration — toutes les slices", fontsize=14, y=1.01)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  Section 7 — RMSE comparison                                                  #
# --------------------------------------------------------------------------- #

def plot_rmse(summary: pd.DataFrame, beta: float) -> None:
    """
    Print the RMSE table sorted best → worst, then show a heatmap + bar chart.
    """
    rmse_df = summary[["expiry", "tenor", "rmse_%"]].copy()
    rmse_df["_exp_y"] = rmse_df["expiry"].str.replace("y", "").astype(float)
    rmse_df["_ten_y"] = rmse_df["tenor"].str.replace("y", "").astype(float)
    rmse_df = (
        rmse_df.sort_values("rmse_%")
        .drop(columns=["_exp_y", "_ten_y"])
        .reset_index(drop=True)
    )
    rmse_df.index += 1

    print("RMSE par slice (du meilleur fit au moins bon) :")
    print(rmse_df.to_string())
    best_idx  = rmse_df["rmse_%"].idxmin()
    worst_idx = rmse_df["rmse_%"].idxmax()
    print(f"\nMoyenne : {rmse_df['rmse_%'].mean():.4f}%")
    print(
        f"Max     : {rmse_df['rmse_%'].max():.4f}%  "
        f"({rmse_df.loc[worst_idx, 'expiry']} x {rmse_df.loc[worst_idx, 'tenor']})"
    )
    print(
        f"Min     : {rmse_df['rmse_%'].min():.4f}%  "
        f"({rmse_df.loc[best_idx, 'expiry']} x {rmse_df.loc[best_idx, 'tenor']})"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Heatmap ──────────────────────────────────────────────────────────── #
    ax1 = axes[0]
    pivot_rmse = sort_index_numeric(
        summary.pivot(index="expiry", columns="tenor", values="rmse_%")
    )
    im = ax1.imshow(pivot_rmse.values, aspect="auto", cmap="RdYlGn_r")
    ax1.set_xticks(range(len(pivot_rmse.columns)))
    ax1.set_xticklabels(pivot_rmse.columns, rotation=45)
    ax1.set_yticks(range(len(pivot_rmse.index)))
    ax1.set_yticklabels(pivot_rmse.index)
    ax1.set_xlabel("Tenor", fontsize=11)
    ax1.set_ylabel("Expiry", fontsize=11)
    ax1.set_title("Heatmap RMSE (%)", fontsize=11)
    plt.colorbar(im, ax=ax1)

    for i in range(len(pivot_rmse.index)):
        for j in range(len(pivot_rmse.columns)):
            val = pivot_rmse.values[i, j]
            if not np.isnan(val):
                ax1.text(
                    j, i, f"{val:.3f}%",
                    ha="center", va="center",
                    fontsize=8, color="black", fontweight="bold",
                )

    # ── Bar chart ─────────────────────────────────────────────────────────── #
    ax2 = axes[1]
    labels_bar = [f"{r['expiry']}×{r['tenor']}" for _, r in rmse_df.iterrows()]
    values_bar = rmse_df["rmse_%"].values
    bar_colors = [
        "#d73027" if v == values_bar.max()
        else "#1a9850" if v == values_bar.min()
        else "#74add1"
        for v in values_bar
    ]

    bars = ax2.barh(labels_bar, values_bar, color=bar_colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, values_bar):
        ax2.text(
            val + values_bar.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}%",
            va="center", fontsize=9,
        )
    ax2.set_xlabel("RMSE (%)", fontsize=11)
    ax2.set_title(
        "RMSE par slice — du meilleur au moins bon\n"
        "(vert = meilleur fit, rouge = moins bon fit)",
        fontsize=11,
    )
    ax2.set_xlim(0, values_bar.max() * 1.18)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.invert_yaxis()

    fig.suptitle(
        f"Qualité de la calibration SABR (β={beta:.1f}) — RMSE par slice",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  Section 8 — individual smile                                                 #
# --------------------------------------------------------------------------- #

def plot_individual_smile(
    surface,
    vol_surface: pd.DataFrame,
    expiry: str,
    tenor: str,
) -> None:
    """
    Plot a single SABR smile vs. market quotes for one (expiry, tenor) slice.
    Also prints the forward rate and calibrated parameters.
    """
    T   = _parse_tenor(expiry)
    F   = surface.forward_rates[(expiry, tenor)]
    p   = surface.results[(expiry, tenor)]

    mkt = vol_surface[
        (vol_surface["expiry"] == expiry) &
        (vol_surface["tenor"]  == tenor)
    ].copy()
    mkt["K"] = F + mkt["strike_spread_bps"] / 10_000.0

    K_grid     = np.linspace(mkt["K"].min() * 0.97, mkt["K"].max() * 1.03, 300)
    model_vols = sabr_vol_vec(F, K_grid, T, p.alpha, p.beta, p.rho, p.nu)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_grid * 100, model_vols * 100, "b-", lw=2, label="SABR model")
    ax.scatter(
        mkt["K"] * 100, mkt["implied_vol"] * 100,
        color="red", zorder=5, s=60, label="Marché",
    )
    ax.axvline(F * 100, ls="--", color="grey", lw=1, label=f"ATM F={F:.3%}")
    ax.set_xlabel("Strike (%)", fontsize=11)
    ax.set_ylabel("Implied Vol (%)", fontsize=11)
    ax.set_title(
        f"SABR smile — Expiry {expiry}, Tenor {tenor}\n"
        f"α={p.alpha:.5f}  β={p.beta:.2f}  ρ={p.rho:+.4f}  ν={p.nu:.5f}  "
        f"RMSE={p.rmse * 100:.4f}%",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    print(f"\nForward rate F = {F:.4%}")
    print(
        f"Paramètres SABR : alpha={p.alpha:.6f}, beta={p.beta}, "
        f"rho={p.rho:.6f}, nu={p.nu:.6f}"
    )
    print(f"RMSE            : {p.rmse * 100:.4f}%")


# --------------------------------------------------------------------------- #
#  Section 10 — price / vega / smile vs. strike                                #
# --------------------------------------------------------------------------- #

def plot_price_vega_smile(
    surface,
    vol_surface: pd.DataFrame,
    yc,
    expiry: str,
    tenor: str,
    strike_bps: int,
    notional: float,
    option_type: str,
) -> None:
    """
    3-panel figure showing the swaption price, vega, and SABR smile
    as a function of strike (ATM ± 200 bps).
    """
    T_pr   = _parse_tenor(expiry)
    ten_pr = _parse_tenor(tenor)
    F_pr   = yc.swap_rate(T_pr, ten_pr)
    A_pr   = yc.annuity(T_pr, ten_pr)
    p_pr   = surface.results[(expiry, tenor)]

    # ── Implied vol at the selected strike ─────────────────────────────────── #
    K_pr = F_pr + strike_bps / 10_000.0
    sigma_pr = sabr_vol_vec(
        F_pr, np.array([K_pr]), T_pr,
        p_pr.alpha, p_pr.beta, p_pr.rho, p_pr.nu,
    )[0]

    # ── Grids ─────────────────────────────────────────────────────────────── #
    bps_grid = np.linspace(-200, 200, 200)
    K_grid   = F_pr + bps_grid / 10_000.0

    prices_grid = []
    vegas_grid  = []
    vols_grid   = []

    for K_i in K_grid:
        if K_i <= 0:
            prices_grid.append(np.nan)
            vegas_grid.append(np.nan)
            vols_grid.append(np.nan)
            continue
        sig_i = sabr_vol_vec(
            F_pr, np.array([K_i]), T_pr,
            p_pr.alpha, p_pr.beta, p_pr.rho, p_pr.nu,
        )[0]
        price_i = swaption_price(F_pr, K_i, sig_i, T_pr, A_pr, notional, option_type)
        greek_i = black_greeks(F_pr, K_i, sig_i, T_pr, A_pr, notional, option_type)
        prices_grid.append(price_i)
        vegas_grid.append(greek_i["vega_1pct"])
        vols_grid.append(sig_i)

    prices_grid = np.array(prices_grid)
    vegas_grid  = np.array(vegas_grid)
    vols_grid   = np.array(vols_grid)

    # ── Plot ──────────────────────────────────────────────────────────────── #
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(bps_grid, prices_grid / 1000, "b-", lw=2)
    axes[0].axvline(
        strike_bps, ls="--", color="red", lw=1.2,
        label=f"Strike sélectionné ({strike_bps:+d} bps)",
    )
    axes[0].axvline(0, ls=":", color="grey", lw=1, label="ATM")
    axes[0].set_xlabel("Strike vs ATM (bps)")
    axes[0].set_ylabel("Prix (k€)")
    axes[0].set_title(f"Prix du swaption {option_type}\n{expiry} × {tenor}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bps_grid, vegas_grid / 1000, "g-", lw=2)
    axes[1].axvline(strike_bps, ls="--", color="red", lw=1.2)
    axes[1].axvline(0, ls=":", color="grey", lw=1)
    axes[1].set_xlabel("Strike vs ATM (bps)")
    axes[1].set_ylabel("Vega (k€ / +1% vol)")
    axes[1].set_title("Vega en fonction du strike")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(bps_grid, vols_grid * 100, "purple", lw=2)
    axes[2].axvline(
        strike_bps, ls="--", color="red", lw=1.2,
        label=f"σ = {sigma_pr:.2%}",
    )
    axes[2].axvline(0, ls=":", color="grey", lw=1)
    axes[2].set_xlabel("Strike vs ATM (bps)")
    axes[2].set_ylabel("Vol implicite SABR (%)")
    axes[2].set_title("Smile SABR utilisé pour le pricing")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Swaption {option_type.upper()} — {expiry}×{tenor}  "
        f"| F={F_pr:.3%}  A={A_pr:.4f}  Notionnel={notional:,.0f}€",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  Section 12 — beta sensitivity comparison                                     #
# --------------------------------------------------------------------------- #

def plot_beta_comparison(
    vol_surface: pd.DataFrame,
    yc,
    expiry: str,
    tenor: str,
    betas: list | None = None,
    n_restarts: int = 8,
) -> None:
    """
    Calibrate SABR for several values of β on the same slice and compare:
      - Left panel  : implied vol smiles
      - Right panel : residuals (model − market)
    """
    if betas is None:
        betas = [0.0, 0.5, 1.0]

    colors = ["steelblue", "darkorange", "seagreen"]
    labels = [f"β = {b:.1f}" for b in betas]

    T_b   = _parse_tenor(expiry)
    ten_b = _parse_tenor(tenor)
    F_b   = yc.swap_rate(T_b, ten_b)

    mkt_b = vol_surface[
        (vol_surface["expiry"] == expiry) &
        (vol_surface["tenor"]  == tenor)
    ].copy()
    mkt_b["K"] = F_b + mkt_b["strike_spread_bps"] / 10_000.0
    mkt_b = mkt_b.dropna(subset=["implied_vol"])
    mkt_b = mkt_b[mkt_b["implied_vol"] > 0]

    strikes_b     = mkt_b["K"].values
    market_vols_b = mkt_b["implied_vol"].values
    weights_b     = np.where(mkt_b["strike_spread_bps"].values == 0, 3.0, 1.0)

    print(f"Calibration sur la slice {expiry} × {tenor}  (F = {F_b:.4%})\n")
    print(f"{'Beta':<8} {'Alpha':>10} {'Rho':>10} {'Nu':>10} {'RMSE':>10}")
    print("-" * 52)

    params_list = []
    for b in betas:
        p = calibrate_sabr(
            F_b, T_b, strikes_b, market_vols_b,
            beta=b, weights=weights_b, n_restarts=n_restarts,
        )
        params_list.append(p)
        print(
            f"  {b:<6} {p.alpha:>10.6f} {p.rho:>10.6f} "
            f"{p.nu:>10.6f} {p.rmse * 100:>9.4f}%"
        )

    bps_grid  = np.linspace(-200, 200, 300)
    K_grid_b  = F_b + bps_grid / 10_000.0
    K_grid_b  = K_grid_b[K_grid_b > 0]
    bps_plot  = (K_grid_b - F_b) * 10_000
    bps_mkt   = (strikes_b - F_b) * 10_000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left : smiles ───────────────────────────────────────────────────────── #
    ax = axes[0]
    for p, color, label in zip(params_list, colors, labels):
        vols = sabr_vol_vec(F_b, K_grid_b, T_b, p.alpha, p.beta, p.rho, p.nu)
        ax.plot(bps_plot, vols * 100, color=color, lw=2, label=label)

    ax.scatter(
        bps_mkt, market_vols_b * 100,
        color="red", zorder=6, s=60, label="Marché", marker="x", linewidths=2,
    )
    ax.axvline(0, ls=":", color="grey", lw=1)
    ax.set_xlabel("Strike vs ATM (bps)", fontsize=11)
    ax.set_ylabel("Vol implicite (%)", fontsize=11)
    ax.set_title(f"Smile SABR — {expiry} × {tenor}\nF = {F_b:.4%}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right : residuals ───────────────────────────────────────────────────── #
    ax2 = axes[1]
    for p, color, label in zip(params_list, colors, labels):
        vols_at_mkt = sabr_vol_vec(F_b, strikes_b, T_b, p.alpha, p.beta, p.rho, p.nu)
        diff = (vols_at_mkt - market_vols_b) * 100
        ax2.plot(
            bps_mkt, diff, color=color, lw=2, marker="o", ms=5,
            label=f"{label}  (RMSE={p.rmse * 100:.4f}%)",
        )

    ax2.axhline(0, color="red", ls="--", lw=1)
    ax2.axvline(0, ls=":", color="grey", lw=1)
    ax2.set_xlabel("Strike vs ATM (bps)", fontsize=11)
    ax2.set_ylabel("Erreur modèle − marché (%)", fontsize=11)
    ax2.set_title("Résidus par beta", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Impact de β sur la calibration SABR — slice {expiry} × {tenor}",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()
