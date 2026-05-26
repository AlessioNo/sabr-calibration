# SABR Cap & Floor Surface Calibration & Pricing
 
Pipeline complet de calibration et pricing de caps et floors fondé sur le modèle **SABR** (Hagan, Kumar, Léger & Woodward, 2002), appliqué à des données de marché USD réelles du 13 juillet 2016.
 
---
 
## Structure du projet
 
```
sabr-calibration/
├── data/
│   ├── usdois.xlsx            # Courbe OIS USD — 25 piliers (2016→2046)
│   └── cap_vol_surface.xlsx   # Surface vol caps USD — 120 expiries × 19 strikes
├── sabr_calibration_caps_floors.ipynb   # Notebook principal
├── data_loader.py             # Lecture des données de marché
├── yield_curve.py             # Bootstrap courbe OIS
├── sabr.py                    # Modèle SABR + calibration
├── pricer.py                  # Pricing Black-76 + Greeks
├── plot_utils.py              # Visualisations
└── README.md
```
 
---
 
## Pipeline
 
| Étape | Description | Module |
|-------|-------------|--------|
| 1 | Bootstrap de la courbe OIS (interpolation log-linéaire) | `yield_curve.py` |
| 2 | Lecture de la surface de vol caps USD | `data_loader.py` |
| 3 | Calibration SABR slice par slice | `sabr.py` |
| 4 | Pricing Black-76 + Greeks analytiques | `pricer.py` |
| 5 | Analyses : smiles, terme structure, sensibilité β | `plot_utils.py` |
 
---
 
## Données
 
| Fichier | Contenu | Nature |
|---------|---------|--------|
| `usdois.xlsx` | Facteurs d'actualisation OIS USD sur 25 piliers | Données marché réelles |
| `cap_vol_surface.xlsx` | Vols implicites log-normales en % — grille 120 expiries × 19 strikes (0.25%–11%) | Données marché réelles |
 
Les taux forward LIBOR 6M sont **bootstrappés depuis l'OIS** via la relation d'arbitrage exacte — ils ne sont pas synthétiques.
 
---
 
## Modules
 
### `yield_curve.py` — Courbe OIS
 
Construit la courbe à partir des facteurs d'actualisation observés.
 
- `YieldCurve(df)` — interpolation log-linéaire des P(0,T)
- `discount(T)` — facteur d'actualisation P(0,T)
- `forward_rate(T1, T2)` — taux LIBOR 6M forward bootstrappé
- `zero_rate(T)` — taux zéro-coupon continu
- `pv_swap(K, T_start, T_end, delta, notional)` — PV d'un swap payeur (utilisé pour la parité cap–floor)
### `data_loader.py` — Chargement des données
 
- `load_discount_factors(path)` — lit `usdois.xlsx`, retourne un DataFrame `dates/discounts`
- `load_cap_vol_surface(path)` — retourne la surface au format tidy `expiry | strike | implied_vol`
- `load_calibration_smiles(path, yield_curve, expiry_map, K_min, K_max)` — prépare les smiles pour la calibration (forward bootstrappé, filtre zone liquide)
### `sabr.py` — Modèle SABR
 
Implémentation de l'expansion asymptotique de Hagan 2002.
 
- `sabr_vol(F, K, T, alpha, beta, rho, nu)` — vol implicite log-normale (scalaire)
- `sabr_vol_vec(...)` — version vectorisée sur un tableau de strikes
- `shifted_sabr_vol(...)` — Shifted SABR (décalage standard 1% pour taux bas)
- `calibrate_sabr(F, T, strikes, market_vols, beta, shift, n_restarts)` — calibre (α, ρ, ν) pour β fixé par moindres carrés pondérés (L-BFGS-B, pondération ×3 sur le point ATM)
- `calibrate_sabr_beta_free(...)` — calibre les 4 paramètres (α, β, ρ, ν) librement
- `SABRCapSurface` — stocke les paramètres calibrés, interpole linéairement entre les expiries
- `calibrate_surface(smiles, beta, shift, n_restarts)` — calibre toutes les slices
### `pricer.py` — Pricing
 
- `black_caplet(F, K, sigma, T, notional, delta, discount, type_)` — prix d'un caplet/floorlet (Black 1976)
- `cap_price(yield_curve, surface, K, T_mat, ...)` — cap complet = Σ caplets, retourne le prix total et la décomposition par caplet
- `floor_price(...)` — prix par parité Cap − Floor = PV(Swap)
- `cap_greeks(...)` — Greeks agrégés analytiques : delta_bp, vega_1%, gamma, vanna, volga
### `plot_utils.py` — Visualisations
 
| Fonction | Description |
|----------|-------------|
| `plot_yield_curve(yc)` | Taux zéro-coupon et forwards bootstrappés |
| `plot_vol_surface_overview(df, strikes)` | Smiles par expiry + terme structure ATM |
| `plot_sabr_params_term_structure(summary)` | Structure temporelle de α, ρ, ν, RMSE |
| `plot_all_smiles(surface, smiles, ...)` | Grille SABR calibré vs marché |
| `plot_rmse(summary, beta)` | Barplot et distribution des RMSE |
| `plot_individual_smile(surface, smiles, label)` | Zoom sur une smile + résidus |
| `plot_price_vega_smile(yc, surface, label, ...)` | Prix, vega, vol SABR en fonction du strike |
| `plot_parity_verification(yc, surface, T_mat)` | Vérification Cap − Floor = PV(Swap) |
| `plot_beta_comparison(smiles, yc, label)` | Sensibilité au choix de β ∈ {0, 0.5, 1, libre} |
 
---
 
## Paramètres clés
 
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `BETA` | `0.0` | Exposant CEV — 0 = normal (Bachelier), optimal USD 2016 |
| `SHIFT` | `0.01` | Décalage Shifted SABR (1% = 100 bps, standard marché) |
| `DELTA` | `0.5` | Fréquence des caplets (6M) |
| `K_MIN` | `0.005` | Borne inférieure de la zone liquide de calibration |
| `K_MAX` | `0.06` | Borne supérieure de la zone liquide de calibration |
| `RESTARTS` | `5` | Nombre de restarts aléatoires par slice |
 
---
 
## Concepts clés
 
**Modèle SABR** — Le forward F et sa volatilité α suivent des processus stochastiques corrélés (corrélation ρ). Le paramètre β contrôle le backbone du smile (dépendance de σ_ATM au niveau de F), et ν est la vol of vol qui pilote les ailes.
 
**Shifted SABR** — En environnement de taux bas (USD 2016 : F ≈ 0.6%), un décalage de 1% sur F et K stabilise la calibration et permet de traiter des strikes proches de zéro sans singularité.
 
**Parité Cap–Floor** — Seule la surface de vol caps est calibrée. Les floors sont déduits par arbitrage : `Floor(K) = Cap(K) − PV(Swap payeur K)`, en utilisant uniquement la courbe OIS.
 
**β = 0 optimal sur USD 2016** — Le régime de taux très bas rend le modèle normal (Bachelier) plus adapté que le log-normal (β=1) : β libre converge vers 0–0.3, réduisant le RMSE de ~88% par rapport à β=1.
 
---
 
## Dépendances
 
```
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
from data_loader import load_discount_factors, load_calibration_smiles
from yield_curve  import YieldCurve
from sabr         import calibrate_surface
from pricer       import cap_price, floor_price, cap_greeks
 
BASE_DIR = Path(".")
yc = YieldCurve(load_discount_factors(BASE_DIR / "data" / "usdois.xlsx"))
 
EXPIRY_MAP = {'1Y':1,'2Y':2,'3Y':3,'5Y':5,'7Y':7,'10Y':10,'15Y':15,'20Y':20,'30Y':30}
smiles  = load_calibration_smiles(BASE_DIR / "data" / "cap_vol_surface.xlsx",
                                   yc, EXPIRY_MAP, K_min=0.005, K_max=0.06)
surface = calibrate_surface(smiles, beta=0.0, shift=0.01)
surface.build_interpolators()
 
# Prix d'un cap 5Y strike 2%, notionnel 1M USD
prix, df = cap_price(yc, surface, K=0.02, T_mat=5.0)
print(f"Cap 5Y 2% : {prix:,.0f} $")
 
# Greeks
g = cap_greeks(yc, surface, K=0.02, T_mat=5.0)
print(f"Delta : {g['delta_bp']:.1f} $/bp  |  Vega : {g['vega_1pct']:.1f} $/1%vol")
```
 
---
 
## Référence
 
Hagan, P., Kumar, D., Léger, A., Woodward, D. (2002). *Managing Smile Risk*. Wilmott Magazine.
 












