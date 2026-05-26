# SABR Swaption Surface Calibration

Calibration SABR complète (Hagan et al. 2002) à partir de discount factors de marché et de cotations de swaptions.

## Structure

```
sabr_calibration/
├── data/
│   ├── market_discount_factors.xlsx   ← discount factors P(0,T)
│   └── swaption_quotes.xlsx           ← vols ATM + spreads par strike
├── data_loader.py    ← lecture et reconstruction de la surface de vol
├── yield_curve.py    ← bootstrap de la courbe (interpolation log-linéaire)
├── sabr.py           ← formule Hagan 2002 + calibration + SABRSurface
├── main.py           ← pipeline principal (CLI)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Calibration simple (affiche les paramètres)
python main.py

# Avec génération de graphiques (dossier plots/)
python main.py --plot

# Paramètres personnalisables
python main.py --beta 0.5 --restarts 10 --plot

# Chemins alternatifs
python main.py --df-path /chemin/vers/discount_factors.xlsx \
               --sw-path /chemin/vers/swaption_quotes.xlsx
```

## Données

### market_discount_factors.xlsx
- Colonne `Date` : dates des pilliers
- Colonne `P(0,T)` : facteurs d'actualisation

### swaption_quotes.xlsx — feuille "swaption ATM vol"
- Matrice expiry × tenor de vols ATM (valeurs absolues, ex: 0.215 = 21.5%)

### swaption_quotes.xlsx — feuille "swaption vol strikes"
- Colonnes `Expiry` et `Tenor` (uniquement les combinaisons avec smile complet)
- Colonnes `-200, -100, -50, -25, 25, 50, 100, 200` : **différences** par rapport à la vol ATM en points de vol (ex: 0.0871 = +8.71 vol points)
- Vol absolue = vol_ATM(expiry, tenor) + spread(strike)

## Méthodologie

### Courbe des taux (`yield_curve.py`)
- Interpolation log-linéaire des P(0,T) sur les piliers de marché
- Taux swap forward : `S(T_start, tenor) = (P(T_start) - P(T_end)) / annuity`
- Annuité calculée avec coupons annuels par défaut

### Calibration SABR (`sabr.py`)
- Formule de Hagan et al. (2002) — approximation lognormale
- **Beta fixé** (défaut 0.5) ; calibration de **(alpha, rho, nu)**
- Méthode : moindres carrés pondérés (L-BFGS-B), plusieurs restarts aléatoires
- Pondération x3 sur le point ATM pour garantir le fit au centre du smile

### Slices calibrées
La calibration du smile nécessite au moins 3 strikes distincts.  
Les (expiry, tenor) disponibles dans le fichier de strikes sont :
- Expiries : 5y, 10y, 20y, 30y
- Tenors   : 2y, 5y, 10y, 20y, 30y

Les autres cellules de la matrice ATM (11×9) n'ont pas de données smile → ignorées pour la calibration SABR (un seul point ATM ne suffit pas).

## Outputs

### Console
- Tableau complet des paramètres (alpha, beta, rho, nu, RMSE)
- Pivots par paramètre : expiry × tenor

### Plots (option `--plot`)
- `plots/smile_<expiry>_<tenor>.png` : smile marché vs modèle SABR
- `plots/sabr_heatmaps.png` : heatmaps alpha / rho / nu sur la surface
