# SABR Cap & Floor Surface Calibration

Calibration SABR complète (Hagan et al. 2002) à partir de discount factors de marché OIS
et de cotations de caps, avec pricing de caps et floors par la formule de Black et parité.

## Structure

```
caps_calibration/
├── usdois.xlsx                      ← discount factors P(0,T) USD OIS
├── cap_vol_surface__1_.xlsx         ← surface vol implicite caps USD
├── data_loader.py    ← lecture OIS et surface vol caps
├── yield_curve.py    ← bootstrap courbe (interpolation log-linéaire + forwards)
├── sabr.py           ← formule Hagan 2002 + calibration + SABRCapSurface
├── pricer.py         ← pricing Black-76 caps/floors + Greeks analytiques
├── plot_utils.py     ← tous les graphiques du notebook
├── sabr_calibration_caps_floors.ipynb  ← notebook principal
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage — notebook

```bash
jupyter notebook sabr_calibration_caps_floors.ipynb
```

Modifier les paramètres en tête de notebook :

```python
BETA      = 0.0    # exposant CEV — 0=normal (optimal USD 2016, RMSE -78% vs 0.5)
RESTARTS  = 5      # restarts par slice
SHIFT     = 0.01   # décalage Shifted SABR (1%)
K_MIN     = 0.005  # borne inférieure zone de calibration
K_MAX     = 0.06   # borne supérieure zone de calibration
```

## Données

### usdois.xlsx
- Colonne `dates` : dates des piliers
- Colonne `discounts` : facteurs d'actualisation P(0,T)

### cap_vol_surface__1_.xlsx
- Index : expiries (3M à 30Y — 120 lignes)
- Colonnes : strikes en % (0.25 à 11 — 19 colonnes)
- Valeurs : volatilités implicites log-normales en %

## Méthodologie

### Courbe des taux (`yield_curve.py`)
- Interpolation log-linéaire des P(0,T) sur les 25 piliers OIS
- Taux forward LIBOR 6M : `F(T1,T2) = (P(T1)/P(T2) - 1) / (T2-T1)`
- Bootstrapping = relation d'arbitrage exacte, pas synthétique

### Calibration SABR (`sabr.py`)
- Formule Hagan et al. (2002) — approximation log-normale
- **Shifted SABR** : shift = 1% pour stabiliser les taux bas USD 2016
- **β fixé à 0** (régime normal/Bachelier, optimal pour taux bas USD 2016) ; calibration de **(α, ρ, ν)**
- Méthode : L-BFGS-B avec 5 restarts aléatoires
- Pondération ×3 sur le strike ATM
- Zone de calibration : strikes 0.5% – 6% (ailes extrêmes exclues)
- β = 0 réduit le RMSE de 78% vs convention β = 0.5 (644 → 140 bps)

### Slices calibrées
Toutes les expiries liquides : 1Y, 2Y, 3Y, 4Y, 5Y, 6Y, 7Y, 8Y, 9Y, 10Y, 15Y, 20Y, 25Y, 30Y.

### Pricing (`pricer.py`)
- Cap = Σ caplets (Black-76 avec vol SABR)
- Floor = Cap − PV(Swap) par parité (aucune vol supplémentaire)
- Greeks : Delta, Vega, Gamma, Vanna, Volga en formule fermée

## Structure du notebook (12 sections)

| Section | Contenu |
|---------|---------|
| 1 | Construction courbe OIS + forwards bootstrappés |
| 2 | Chargement et exploration surface vol caps |
| 3 | Calibration SABR — méthode et procédure |
| 4 | Résultats — tableau paramètres calibrés |
| 5 | Smiles calibrées vs marché (toutes expiries) |
| 6 | RMSE par expiry — qualité de la calibration |
| 7 | Smile individuelle avec résidus |
| 8 | Pricing cap avec Greeks et décomposition caplets |
| 9 | Profils prix et vega en fonction du strike |
| 10 | Pricing floor par parité Cap–Floor–Swap |
| 11 | Vérification graphique de la parité |
| 12 | Sensibilité à β ∈ {0, 0.5, 1} |

## Outputs

### Console (pendant la calibration)
- Paramètres calibrés (α, ρ, ν, RMSE) par expiry

### Notebook
- Tableaux stylés avec gradient de couleurs
- Graphiques : smiles, terme structure, RMSE, parité, comparaison β

## Note sur le RMSE

Le RMSE est élevé sur certaines expiries (100–1000+ bps) — ce n'est pas une erreur de code.
La smile USD 2016 est fortement asymétrique (forme en U inclinée) car les taux étaient
à 0.4–1.7% post-GFC. Le SABR standard ne peut pas reproduire cette asymétrie extrême.

Extension naturelle : β libre (proche de 0, régime Bachelier) réduit le RMSE de ~88%.
Voir Section 12 du notebook.
