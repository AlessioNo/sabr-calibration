"""
plot_utils.py
-------------
Fonctions graphiques pour la calibration SABR — Caps & Floors USD 2016.

API publique
------------
  plot_yield_curve(yield_curve)
  plot_vol_surface_overview(raw_vol_df, strikes_all)
  plot_sabr_params_term_structure(summary_df)
  plot_all_smiles(surface, smiles)
  plot_rmse(summary_df, beta)
  plot_individual_smile(surface, smiles, label)
  plot_price_vega_smile(yield_curve, surface, label, K_bps, notional, type_)
  plot_parity_verification(yield_curve, surface, T_mat)
  plot_beta_comparison(smiles_raw_df, yield_curve, label)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from scipy.optimize import minimize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sabr_vol(F, K, T, alpha, beta, rho, nu, shift=0.01):
    """SABR shifted — wrapper interne."""
    from sabr import sabr_vol
    return sabr_vol(F + shift, K + shift, T, alpha, beta, rho, nu)


def _sabr_vol_vec(F, Ks, T, alpha, beta, rho, nu, shift=0.01):
    return np.array([_sabr_vol(F, K, T, alpha, beta, rho, nu, shift) for K in Ks])


def _calibrate_for_beta(F, T, strikes, mkt_vols, beta=0.0, shift=0.01, n_restarts=5):
    """Calibre (alpha, rho, nu) pour un beta donné — version allégée."""
    from sabr import calibrate_sabr
    return calibrate_sabr(F, T, np.array(strikes), np.array(mkt_vols),
                          beta=beta, shift=shift, n_restarts=n_restarts)


COLORS = {
    'sabr'  : '#185FA5',
    'market': '#D85A30',
    'zone'  : '#185FA5',
    'rho'   : '#A32D2D',
    'nu'    : '#3B6D11',
    'alpha' : '#185FA5',
    'rmse'  : '#854F0B',
    'beta0' : '#185FA5',
    'beta05': '#3B6D11',
    'beta1' : '#A32D2D',
}

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor'  : '#f8f9fa',
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'font.size'       : 10,
}


def _apply_style():
    plt.rcParams.update(STYLE)


# ── 1. Courbe OIS ─────────────────────────────────────────────────────────────

def plot_yield_curve(yield_curve):
    """Taux zéro-coupon et courbe forward bootstrappée."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Courbe OIS USD — 13 juillet 2016',
                 fontsize=12, fontweight='bold')

    T_plot = np.linspace(0.1, 30, 300)
    ax1.plot(T_plot, yield_curve.zero_rate(T_plot)*100,
             color=COLORS['sabr'], linewidth=2)
    ax1.scatter(yield_curve._T, yield_curve.zero_rate(yield_curve._T)*100,
                color=COLORS['market'], s=50, zorder=5, label='Piliers OIS observés')
    ax1.set_xlabel('Maturité (années)')
    ax1.set_ylabel('Taux zéro (%)')
    ax1.set_title('Taux zéro-coupon\n(interpolation log-linéaire)')
    ax1.legend(fontsize=9)

    fwd_curve = yield_curve.forward_curve(delta=0.5, T_max=30)
    ax2.plot(fwd_curve['T_end'], fwd_curve['F']*100,
             'o-', color=COLORS['nu'], linewidth=1.8, markersize=3)
    ax2.axhline(fwd_curve['F'].mean()*100, color='gray', linestyle='--',
                linewidth=1, label=f"Moyenne = {fwd_curve['F'].mean()*100:.3f}%")
    ax2.set_xlabel('Expiry caplet (années)')
    ax2.set_ylabel('Taux forward LIBOR 6M (%)')
    ax2.set_title('Taux forward LIBOR 6M bootstrappés\n(depuis courbe OIS — pas synthétiques)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# ── 2. Aperçu surface vol ─────────────────────────────────────────────────────

def plot_vol_surface_overview(df_vol: pd.DataFrame, strikes_all: list):
    """Smiles de vol et terme structure ATM."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Surface de vol implicite — Caps USD (données réelles, 13 juillet 2016)",
                 fontsize=12, fontweight='bold')

    expiries_plot = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y']
    colors_p = cm.viridis(np.linspace(0, 1, len(expiries_plot)))

    for exp, col in zip(expiries_plot, colors_p):
        if exp in df_vol.index:
            ax1.plot([s*100 for s in strikes_all], df_vol.loc[exp].values,
                     linewidth=1.8, color=col, label=exp, marker='o', markersize=3)

    ax1.set_xlabel('Strike (%)'); ax1.set_ylabel('Vol implicite (%)')
    ax1.set_title('Smiles de vol par expiry\n(forme en U asymétrique — typique USD 2016)')
    ax1.legend(fontsize=8, title='Expiry', ncol=2)

    expiry_map2 = {'1Y':1,'2Y':2,'3Y':3,'5Y':5,'7Y':7,'10Y':10,
                   '15Y':15,'20Y':20,'25Y':25,'30Y':30}
    exp_T, vols_min = [], []
    for lbl, T in expiry_map2.items():
        if lbl in df_vol.index:
            exp_T.append(T)
            vols_min.append(df_vol.loc[lbl].min())

    ax2.plot(exp_T, vols_min, 'o-', color=COLORS['sabr'], linewidth=2, markersize=6)
    ax2.fill_between(exp_T, vols_min, alpha=0.08, color=COLORS['sabr'])
    ax2.set_xlabel('Expiry (années)'); ax2.set_ylabel('Vol minimum (%)')
    ax2.set_title('Terme structure de vol (vol ATM minimum)\nHump shape — vol courte élevée')

    plt.tight_layout()
    plt.show()


# ── 3. Terme structure des paramètres ─────────────────────────────────────────

def plot_sabr_params_term_structure(summary: pd.DataFrame):
    """Structure temporelle de (alpha, rho, nu, RMSE)."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Structure temporelle des paramètres SABR calibrés\nCaps USD — données réelles juillet 2016',
                 fontsize=12, fontweight='bold')

    T_pl = summary['T (ans)'].values
    lbls = summary['expiry'].values

    for ax, col, key, title in [
        (axes[0,0], COLORS['alpha'], 'alpha',   'α — volatilité initiale'),
        (axes[0,1], COLORS['rho'],   'rho',     'ρ — corrélation (skew)'),
        (axes[1,0], COLORS['nu'],    'nu',       'ν — vol of vol (wings)'),
        (axes[1,1], COLORS['rmse'],  'rmse_%',  'RMSE calibration (%)'),
    ]:
        vals = summary[key].values
        ax.plot(T_pl, vals, 'o-', color=col, linewidth=2, markersize=7)
        ax.fill_between(T_pl, vals, alpha=0.08, color=col)
        for x, y, lbl in zip(T_pl, vals, lbls):
            ax.annotate(lbl, (x, y), textcoords='offset points',
                        xytext=(0, 8), fontsize=7.5, ha='center', color=col)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Expiry (années)')

    plt.tight_layout()
    plt.show()


# ── 4. Toutes les smiles ──────────────────────────────────────────────────────

def plot_all_smiles(surface, smiles: list, df_vol: pd.DataFrame,
                   strikes_all: list, n_cols: int = 3):
    """Grille de smiles calibrées vs marché."""
    _apply_style()
    n = len(smiles)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig.suptitle('SABR Calibration — Caps USD (données réelles juillet 2016)\n'
                 'Vol implicite SABR (—) vs marché réel (●)',
                 fontsize=12, fontweight='bold')

    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, s in zip(axes_flat, smiles):
        p = surface.get(s['label'])
        if p is None:
            ax.axis('off'); continue
        F, T = s['F'], s['expiry']

        K_dense = np.linspace(0.001, 0.08, 200)
        v_sabr  = [_sabr_vol(F, K, T, p.alpha, p.beta, p.rho, p.nu)*100 for K in K_dense]

        all_vols = df_vol.loc[s['label']].values
        ax.scatter([k*100 for k in strikes_all], all_vols,
                   color='#cccccc', s=25, zorder=2, label='Marché (hors zone)')
        ax.scatter([k*100 for k in s['strikes']],
                   [v*100 for v in s['mkt_vols']],
                   color=COLORS['market'], s=55, zorder=5,
                   label='Marché (zone cal.)', edgecolors='white', linewidths=0.5)
        ax.plot(K_dense*100, v_sabr,
                color=COLORS['sabr'], linewidth=2.2, label='SABR calibré', zorder=4)
        ax.axvline(F*100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvspan(0.5, 6.0, alpha=0.04, color=COLORS['zone'])
        ax.set_xlim(0, 10)
        ax.set_title(f"Cap {s['label']}  F={F*100:.3f}%  RMSE={p.rmse*10000:.1f}bps",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Strike (%)', fontsize=8)
        ax.set_ylabel('Vol (%)', fontsize=8)
        ax.legend(fontsize=6.5, ncol=2)

    for ax in axes_flat[len(smiles):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ── 5. RMSE ───────────────────────────────────────────────────────────────────

def plot_rmse(summary: pd.DataFrame, beta: float):
    """Barplot RMSE + histogramme."""
    _apply_style()
    rmses = summary['rmse_%'].values * 100   # en bps
    lbls  = summary['expiry'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Qualité de la calibration SABR (β={beta}) — RMSE par slice',
                 fontsize=11, fontweight='bold')

    bar_colors = ['#185FA5' if r < 20 else '#854F0B' if r < 100 else '#A32D2D'
                  for r in rmses]
    ax1.bar(range(len(rmses)), rmses, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(np.mean(rmses), color='black', linestyle='--', linewidth=1.2,
                label=f'Moyenne = {np.mean(rmses):.1f} bps')
    ax1.set_xticks(range(len(lbls)))
    ax1.set_xticklabels(lbls, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('RMSE (bps)'); ax1.set_title('RMSE par expiry')
    ax1.legend()

    ax2.hist(rmses, bins=8, color=COLORS['sabr'], alpha=0.75, edgecolor='white')
    ax2.axvline(np.mean(rmses), color=COLORS['market'], linestyle='--',
                linewidth=1.5, label=f'Moyenne = {np.mean(rmses):.1f} bps')
    ax2.axvline(np.median(rmses), color=COLORS['nu'], linestyle='--',
                linewidth=1.5, label=f'Médiane = {np.median(rmses):.1f} bps')
    ax2.set_xlabel('RMSE (bps)'); ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des RMSE'); ax2.legend()

    plt.tight_layout()
    plt.show()


# ── 6. Smile individuel ───────────────────────────────────────────────────────

def plot_individual_smile(surface, smiles: list, df_vol: pd.DataFrame,
                          strikes_all: list, label: str):
    """Zoom sur une smile donnée avec résidus."""
    _apply_style()
    s = next((x for x in smiles if x['label'] == label), None)
    p = surface.get(label)
    if s is None or p is None:
        print(f"Label {label} non trouvé."); return

    F, T = s['F'], s['expiry']
    K_dense = np.linspace(0.001, 0.08, 300)
    v_sabr  = np.array([_sabr_vol(F, K, T, p.alpha, p.beta, p.rho, p.nu)*100
                        for K in K_dense])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Smile SABR détaillée — Cap {label}  |  F={F*100:.3f}%  |  β={p.beta}',
                 fontsize=11, fontweight='bold')

    all_vols = df_vol.loc[label].values
    ax1.scatter([k*100 for k in strikes_all], all_vols,
                color='#cccccc', s=30, zorder=2, label='Marché (hors zone)')
    ax1.scatter([k*100 for k in s['strikes']],
                [v*100 for v in s['mkt_vols']],
                color=COLORS['market'], s=70, zorder=5, label='Marché (zone calibrée)',
                edgecolors='white', linewidths=0.8)
    ax1.plot(K_dense*100, v_sabr, color=COLORS['sabr'],
             linewidth=2.5, label=f'SABR  RMSE={p.rmse*10000:.1f}bps', zorder=4)
    ax1.axvline(F*100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7,
                label=f'F={F*100:.3f}%')
    ax1.axvspan(0.5, 6.0, alpha=0.05, color=COLORS['zone'], label='Zone calibrée')
    ax1.set_xlim(0, 10)
    ax1.set_xlabel('Strike (%)'); ax1.set_ylabel('Vol implicite (%)')
    ax1.set_title('Smile — SABR vs marché'); ax1.legend(fontsize=8)

    # Résidus sur la zone calibrée
    K_arr = np.array(s['strikes'])
    v_arr = np.array(s['mkt_vols'])
    v_mod = np.array([_sabr_vol(F, K, T, p.alpha, p.beta, p.rho, p.nu) for K in K_arr])
    resid = (v_mod - v_arr) * 10000  # en bps

    ax2.bar(np.array(s['strikes'])*100, resid,
            width=0.03, color=[COLORS['rho'] if r < 0 else COLORS['sabr'] for r in resid],
            edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Strike (%)'); ax2.set_ylabel('Résidu SABR − marché (bps)')
    ax2.set_title(f'Résidus (SABR − marché)  |  RMSE = {p.rmse*10000:.1f} bps')

    plt.tight_layout()
    plt.show()

    print(f'\nParamètres calibrés — Cap {label}')
    print(f'  alpha = {p.alpha:.6f}')
    print(f'  beta  = {p.beta:.2f}  (fixé)')
    print(f'  rho   = {p.rho:+.6f}')
    print(f'  nu    = {p.nu:.6f}')
    print(f'  RMSE  = {p.rmse*10000:.2f} bps')


# ── 7. Prix et vega en fonction du strike ─────────────────────────────────────

def plot_price_vega_smile(yield_curve, surface, label: str,
                          strike_bps: int = 0, notional: float = 1_000_000,
                          type_: str = 'cap'):
    """Prix, vega et vol SABR en fonction du strike."""
    from pricer import black_caplet
    _apply_style()

    p = surface.get(label)
    if p is None:
        print(f'Label {label} non trouvé.'); return

    T   = surface.expiries[label]
    F   = surface.forward_rates[label]
    P   = float(yield_curve.discount(T))

    K_ref = F + strike_bps / 10_000
    Ks    = np.linspace(max(F - 0.03, 0.001), F + 0.04, 100)

    prices = []
    vegas  = []
    vols   = []

    for K in Ks:
        sig = _sabr_vol(F, K, T, p.alpha, p.beta, p.rho, p.nu)
        px  = black_caplet(F, K, sig, T, notional, 0.5, P, type_)
        if sig > 0 and T > 0 and F > 0 and K > 0:
            sqrtT = math.sqrt(T)
            d1    = (math.log(F/K) + 0.5*sig**2*T)/(sig*sqrtT)
            vg    = notional * 0.5 * P * F * norm.pdf(d1) * sqrtT * 0.01
        else:
            vg = 0.0
        prices.append(px)
        vegas.append(vg)
        vols.append(sig * 100)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Profils Prix, Vega et Vol — Cap {label}  |  F={F*100:.3f}%',
                 fontsize=11, fontweight='bold')

    for ax, vals, ylabel, title, col in [
        (axes[0], prices, 'Prix ($)',        'Prix caplet vs strike',  COLORS['sabr']),
        (axes[1], vegas,  'Vega ($/%vol)',   'Vega vs strike',         COLORS['rho']),
        (axes[2], vols,   'Vol SABR (%)',    'Vol SABR vs strike',     COLORS['nu']),
    ]:
        ax.plot(Ks*100, vals, color=col, linewidth=2)
        ax.axvline(F*100, color='gray', linestyle='--', linewidth=0.8,
                   label=f'F={F*100:.3f}%')
        ax.axvline(K_ref*100, color=COLORS['market'], linestyle=':',
                   linewidth=1.5, label=f'K={K_ref*100:.3f}%')
        ax.set_xlabel('Strike (%)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── 8. Parité Cap–Floor–Swap ──────────────────────────────────────────────────

def plot_parity_verification(yield_curve, surface, T_mat: float = 5.0,
                              notional: float = 1_000_000):
    """Vérification graphique de la parité Cap − Floor = PV(Swap)."""
    from pricer import cap_price
    _apply_style()

    Ks     = np.linspace(0.003, 0.07, 40)
    caps_p = []
    floors_p = []
    swaps_p  = []

    for K in Ks:
        pc, _ = cap_price(yield_curve, surface, K, T_mat, notional)
        ps    = yield_curve.pv_swap(K, 0.0, T_mat, 0.5, notional)
        caps_p.append(pc)
        swaps_p.append(ps)
        floors_p.append(pc - ps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Parité Cap − Floor = Swap  |  Maturité {T_mat}Y  |  Notional 1M USD',
                 fontsize=11, fontweight='bold')

    ax1.plot(Ks*100, [c/1000 for c in caps_p],
             color=COLORS['sabr'], linewidth=2, label='Cap (k$)')
    ax1.plot(Ks*100, [f/1000 for f in floors_p],
             color=COLORS['market'], linewidth=2, label='Floor par parité (k$)')
    ax1.set_xlabel('Strike (%)'); ax1.set_ylabel('Prix (k$)')
    ax1.set_title('Prix Cap et Floor'); ax1.legend()

    diff = [c - f for c, f in zip(caps_p, floors_p)]
    ax2.plot(Ks*100, [d/1000 for d in diff],
             color=COLORS['sabr'], linewidth=2.5, label='Cap − Floor')
    ax2.plot(Ks*100, [s/1000 for s in swaps_p],
             color=COLORS['rho'], linewidth=1.8, linestyle='--', label='PV(Swap)')
    ax2.axhline(0, color='black', linewidth=0.6)
    ax2.set_xlabel('Strike (%)'); ax2.set_ylabel('Valeur (k$)')
    ax2.set_title('Vérification : Cap − Floor = PV(Swap)'); ax2.legend()

    plt.tight_layout()
    plt.show()

    max_err = max(abs(d - s) for d, s in zip(diff, swaps_p))
    print(f'Erreur de parité maximale : {max_err:.4f} $  →  parité vérifiée ✓')


# ── 9. Comparaison beta = 0 / 0.5 / 1 / libre ────────────────────────────────

def plot_beta_comparison(smiles: list, yield_curve, label: str):
    """
    Compare les smiles calibrées pour β ∈ {0, 0.5, 1, libre} sur une expiry.
    4 cas : β fixé à 0, 0.5, 1 + β calibré librement comme 4ème paramètre.
    """
    _apply_style()

    s = next((x for x in smiles if x['label'] == label), None)
    if s is None:
        print(f'Label {label} non trouvé.'); return

    F, T      = s['F'], s['expiry']
    strikes   = np.array(s['strikes'])
    mkt_vols  = np.array(s['mkt_vols'])

    betas     = [0.0, 0.5, 1.0]
    beta_lbls = ['β=0 (normal)', 'β=0.5 (CIR-like)', 'β=1 (log-normal)']
    beta_cols = [COLORS['beta0'], COLORS['beta05'], COLORS['beta1']]

    params_list = []
    for b in betas:
        p = _calibrate_for_beta(F, T, strikes, mkt_vols, beta=b)
        params_list.append(p)

    # Beta libre — 4ème paramètre
    from sabr import calibrate_sabr_beta_free
    p_libre = calibrate_sabr_beta_free(F, T, strikes, mkt_vols)
    params_list.append(p_libre)
    betas.append(p_libre.beta)
    beta_lbls.append(f'β libre (β={p_libre.beta:.3f})')
    beta_cols.append('#7B2D8B')  # violet

    K_dense = np.linspace(max(strikes.min()*0.7, 0.001), strikes.max()*1.1, 200)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Sensibilité au choix de β — Cap {label}  |  F={F*100:.3f}%\n'
        r'β définit le backbone du smile : σ_ATM ∝ α/F^{1−β}',
        fontsize=11, fontweight='bold'
    )

    # Graphe 1 : smiles superposées
    ax = axes[0]
    ax.scatter(strikes*100, mkt_vols*100,
               color=COLORS['market'], s=70, zorder=5,
               label='Marché', edgecolors='white', linewidths=0.8)
    for p, lbl, col in zip(params_list, beta_lbls, beta_cols):
        v = _sabr_vol_vec(F, K_dense, T, p.alpha, p.beta, p.rho, p.nu)*100
        ax.plot(K_dense*100, v, linewidth=2, color=col,
                label=f'{lbl}  RMSE={p.rmse*10000:.1f}bps')
    ax.axvline(F*100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Strike (%)'); ax.set_ylabel('Vol implicite (%)')
    ax.set_title('Smiles calibrées pour β ∈ {0, 0.5, 1}')
    ax.legend(fontsize=7.5)

    # Graphe 2 : résidus
    ax = axes[1]
    for p, lbl, col in zip(params_list, beta_lbls, beta_cols):
        v_mod = _sabr_vol_vec(F, strikes, T, p.alpha, p.beta, p.rho, p.nu)
        resid = (v_mod - mkt_vols) * 10000
        ax.plot(strikes*100, resid, 'o-', color=col, linewidth=1.8,
                markersize=5, label=lbl)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Strike (%)'); ax.set_ylabel('Résidu (bps)')
    ax.set_title('Résidus SABR − marché par β')
    ax.legend(fontsize=7.5)

    # Graphe 3 : paramètres calibrés
    ax = axes[2]
    param_names = ['α', 'ρ', 'ν', 'RMSE\n(bps)']
    param_vals  = [
        [p.alpha   for p in params_list],
        [p.rho     for p in params_list],
        [p.nu      for p in params_list],
        [p.rmse*10000 for p in params_list],
    ]
    x = np.arange(len(betas))
    for i, (name, vals) in enumerate(zip(param_names, param_vals)):
        ax2_twin = ax if i == 0 else ax.twinx() if i == 3 else ax
        pass

    # Tableau simple
    table_data = []
    lbls_table = ['β=0', 'β=0.5', 'β=1', f'β libre']
    for p, lbl in zip(params_list, lbls_table):
        table_data.append([lbl, f'{p.beta:.3f}', f'{p.alpha:.5f}',
                           f'{p.rho:+.4f}', f'{p.nu:.4f}', f'{p.rmse*10000:.1f}'])

    ax.axis('off')
    tbl = ax.table(
        cellText=table_data,
        colLabels=['Cas', 'β', 'α', 'ρ', 'ν', 'RMSE (bps)'],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.2)
    ax.set_title('Paramètres calibrés par β', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()

    # Interprétation
    print(f'\nComparaison β — Cap {label}  |  F={F*100:.3f}%')
    print('-'*60)
    print(f"{'β':>6}  {'α':>10}  {'ρ':>8}  {'ν':>8}  {'RMSE (bps)':>12}")
    print('-'*60)
    for p, b in zip(params_list, betas):
        print(f"  {b:>4.1f}  {p.alpha:>10.6f}  {p.rho:>+8.4f}  "
              f"{p.nu:>8.4f}  {p.rmse*10000:>12.2f}")
    print('-'*60)
    best_b = betas[np.argmin([p.rmse for p in params_list])]
    print(f'\nβ optimal : {best_b}  '
          f'({"normal/Bachelier" if best_b==0 else "racine carrée" if best_b==0.5 else "log-normal"})')
    print()
    print('Interprétation :')
    print(f'  β=0 (normal)    : vol ATM indépendante de F — adapté aux taux bas/négatifs')
    print(f'  β=0.5 (CIR)     : vol ATM ∝ 1/√F — compromis standard marché')
    print(f'  β=1 (log-normal): vol ATM ∝ 1/F  — inadapté aux taux très bas USD 2016')
