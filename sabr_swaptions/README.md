# SABR Swaption Surface Calibration & Pricing

Pipeline complet de calibration et de pricing de swaptions fondé sur le modèle **SABR** (Hagan, Kumar, Léger & Woodward, 2002). Les swaptions (options sur taux de swap) sont modélisées et calibrées à partir de données de marché (facteurs d'actualisation et volatilités implicites).

---

## Structure du projet

```text
sabr-calibration/
├── data/
│   ├── market_discount_factors.xlsx   # Facteurs d'actualisation du marché pour la courbe des taux
│   └── swaption_quotes.xlsx           # Vols ATM et spreads de strikes pour les swaptions
├── sabr_calibration_swaptions.ipynb   # Notebook principal d'analyse
├── data_loader.py             # Lecture et préparation des données de marché
├── yield_curve.py             # Bootstrap de la courbe des taux, annuités et taux forward
├── sabr.py                    # Modèle SABR + calibration de la surface
├── pricer.py                  # Pricing Black-76 + Greeks (Swaptions Payer / Receiver)
├── plot_utils.py              # Helpers de visualisation Matplotlib
└── README.md
```

---

## Pipeline

| Étape | Description | Module |
|-------|-------------|--------|
| 1 | Bootstrap de la courbe des taux à partir des facteurs d'actualisation observés | `yield_curve.py` |
| 2 | Lecture et fusion des volatilités ATM et des spreads de swaptions | `data_loader.py` |
| 3 | Calibration SABR pour chaque couple (Expiry, Tenor) | `sabr.py` |
| 4 | Pricing Black-76 (payer / receveur) + calcul des Greeks analytiques | `pricer.py` |
| 5 | Analyses : smiles de vol, heatmaps, paramètres de surface, sensibilité β | `plot_utils.py` |

---

## Données

| Fichier | Contenu | Nature |
|---------|---------|--------|
| `market_discount_factors.xlsx` | Facteurs d'actualisation P(0,T) associés à des dates spécifiques | Données marché réelles |
| `swaption_quotes.xlsx` | 1. Feuille **swaption ATM vol** : Grille Expiry × Tenor (1y à 30y)<br>2. Feuille **swaption vol strikes** : Différences (spreads) en bps par rapport à l'ATM (-200 à +200 bps) | Données marché réelles |

Les taux forward (Swap rates) F et les annuités A(T,τ) sont **bootstrappés dynamiquement** depuis les facteurs d'actualisation de la courbe des taux via des relations d'arbitrage exactes.

---

## Modules

### `yield_curve.py` — Courbe des Taux
Construit la courbe à partir des facteurs d'actualisation observés pour interpoler et déduire les taux paratonnerres.

- `YieldCurve(df)` — Interpolation log-linéaire des P(0,T)
- `annuity(expiry, tenor, freq)` — Calcule le PV01 (Annuité) du swap
- `swap_rate(expiry, tenor, freq)` — Calcule le taux de swap forward (F)
- `pillar_summary()` — Génère une table des piliers (T, P(0,T), taux zéro-coupon)

### `data_loader.py` — Chargement des données
Traite les données Excel brutes et construit des structures exploitables.

- `load_discount_factors(path)` — Lit `market_discount_factors.xlsx`, retourne un DataFrame chronologique
- `load_atm_vols(path)` — Lit la matrice des volatilités à la monnaie (ATM)
- *(Interne)* — Fusionne les spreads de volatilité avec la volatilité ATM pour générer des volatilités absolues `implied_vol = atm_vol + vol_spread`

### `sabr.py` — Modèle SABR
Implémentation de l'expansion asymptotique de Hagan (2002).

- `sabr_vol(F, K, T, alpha, beta, rho, nu)` — Formule analytique de volatilité log-normale
- `calibrate_sabr(...)` — Calibre (α, ρ, ν) pour un β fixé par moindres carrés (optimisation avec pondération ×3 sur la vol ATM pour garantir l'ancrage)
- `SABRSurface` — Objet gérant la calibration et l'interpolation sur l'intégralité de la grille (Expiry, Tenor)

### `pricer.py` — Pricing & Greeks
Mécanique de valorisation via le modèle de Black (1976).

- `black_price(...)` — Prix d'une swaption en fraction d'annuité
- `swaption_price(F, K, sigma, T, annuity, notional, option_type)` — Prix complet en devise (payer/receiver)
- `black_greeks(...)` — Greeks de la swaption par rapport au forward F et à la vol implicite : `delta_bp`, `vega_1pct`, `gamma_bp2`, `vanna`, `volga`

### `plot_utils.py` — Visualisations
Encapsule la logique Matplotlib pour le notebook principal.

| Fonction | Description |
|----------|-------------|
| `plot_sabr_params_tables(...)` | Affichage tabulaire des paramètres calibrés |
| `plot_sabr_heatmaps(...)` | Heatmaps de α, ρ, ν sur la grille Expiry × Tenor |
| `plot_all_smiles(...)` | Superposition des smiles calibrés vs points de marché |
| `plot_individual_smile(...)` | Zoom sur un point de la grille + analyse des résidus |
| `plot_price_vega_smile(...)` | Évolution du prix et du vega selon le strike pour un couple défini |
| `plot_beta_comparison(...)` | Sensibilité de la surface au choix de β |

---

## Paramètres clés

| Paramètre | Valeur Typique | Description |
|-----------|--------|-------------|
| `BETA` | `0.5` ou `1.0` | Exposant CEV contrôlant le backbone. Standardisé souvent à 0.5 sur les taux, ou 1.0 (lognormal strict). |
| `WEIGHT_ATM` | `3.0` | Coefficient appliqué au strike spread = 0 (ATM) lors de la calibration pour prioriser l'ajustement à la monnaie. |
| `N_RESTARTS` | `n` | Nombre d'initialisations aléatoires dans l'optimisation pour éviter les minimums locaux. |

---

## Concepts clés

**Modèle SABR** — Le taux forward F et sa volatilité α suivent des processus stochastiques corrélés (corrélation ρ). Le paramètre β contrôle le "backbone" du smile, tandis que ν (vol of vol) dicte la courbure des ailes.

**Swaption Payer / Receiver** — Une swaption "Payer" (Call) donne le droit de payer le taux fixe K et de recevoir le taux variable. Une swaption "Receiver" (Put) donne le droit inverse. Leur valeur monétaire finale nécessite de multiplier le prix Black (1976) par l'**Annuité** A(T,τ) et le Notionnel.

**Cotation en "Spread"** — Dans `swaption_quotes.xlsx`, seule la volatilité ATM (À la monnaie) est donnée en absolu. Les strikes en dehors de la monnaie (OTM/ITM) sont cotés en différences (bps) depuis cette volatilité de base pour former le smile de volatilité complet.

---

## Dépendances

```bash
numpy
pandas
scipy
matplotlib
openpyxl
```

Installation :

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## Utilisation rapide

```python
from pathlib import Path
from data_loader import load_discount_factors
from yield_curve import YieldCurve
from pricer import swaption_price, black_greeks

BASE_DIR = Path(".")

# 1. Charger la courbe des taux
df_discount = load_discount_factors(BASE_DIR / "data" / "market_discount_factors.xlsx")
yc = YieldCurve(df_discount)

# Paramètres du contrat
expiry, tenor = "5y", "10y"
T_years = 5.0
F = yc.swap_rate(expiry, tenor)
annuity = yc.annuity(expiry, tenor)

# 2. Volatilité (issue de votre objet SABRSurface préalablement calibré)
# sigma = surface.sabr_vol(F, K=F, T=T_years)
sigma = 0.172  # Exemple ATM vol pour 5y10y

# 3. Pricing & Greeks (Swaption Payer ATM, notionnel 10M)
notional = 10_000_000
prix = swaption_price(F, K=F, sigma=sigma, T=T_years, annuity=annuity, 
                      notional=notional, option_type="payer")

print(f"Prix de la Swaption Payer 5y10y (ATM) : {prix:,.2f}")

greeks = black_greeks(F, K=F, sigma=sigma, T=T_years, annuity=annuity, 
                      notional=notional, option_type="payer")

print(f"Delta: {greeks['delta_bp']:,.2f} / bp | Vega: {greeks['vega_1pct']:,.2f} / 1% vol")
```

---

## Référence

Hagan, P., Kumar, D., Léger, A., Woodward, D. (2002). *Managing Smile Risk*. Wilmott Magazine.
