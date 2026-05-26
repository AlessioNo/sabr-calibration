# SABR Interest Rate Derivatives : Calibration & Pricing

Ce dépôt regroupe deux projets indépendants de calibration et de pricing d'options sur taux d'intérêt fondés sur le modèle de volatilité stochastique **SABR** (Hagan, Kumar, Léger & Woodward, 2002). Le projet est divisé en deux modules distincts selon l'instrument traité.

---

## 🏗 Structure du projet

Le dépôt est organisé en deux dossiers autonomes contenant chacun leur propre pipeline de données, de calibration et de pricing adaptés aux spécificités de chaque marché :

```text
sabr-calibration/
│
├── sabr_caps_floors/                  # 📌 MODULE CAPS & FLOORS
│   ├── data/
│   │   ├── usdois.xlsx                # Courbe OIS USD — 25 piliers (2016)
│   │   └── cap_vol_surface.xlsx       # Surface vol caps — 120 expiries × 19 strikes
│   ├── sabr_calibration_caps_floors.ipynb # Notebook principal d'analyse
│   ├── data_loader.py                 # Chargement des données Caps
│   ├── yield_curve.py                 # Bootstrap courbe OIS & forwards LIBOR 6M
│   ├── sabr.py                        # Modèle SABR (Shifted SABR, bêta fixe/libre)
│   ├── pricer.py                      # Pricing Black-76 + Greeks analytiques
│   └── plot_utils.py                  # Visualisations des smiles et structures à terme
│
├── sabr_swaptions/                    # 📌 MODULE SWAPTIONS
│   ├── data/
│   │   ├── market_discount_factors.xlsx # Facteurs d'actualisation du marché
│   │   └── swaption_quotes.xlsx       # Matrice vols ATM et spreads de strikes
│   ├── sabr_calibration_swaptions.ipynb # Notebook principal d'analyse
│   ├── data_loader.py                 # Reconstruction de la surface via les spreads
│   ├── yield_curve.py                 # Interpolation log-linéaire, Swap rates & Annuités
│   ├── sabr.py                        # Modèle SABR avec moindres carrés pondérés ATM
│   ├── pricer.py                      # Pricing de Swaptions (Payer/Receiver) + Greeks
│   └── plot_utils.py                  # Heatmaps des paramètres et smiles de swaptions
│
└── README.md                          # Ce fichier
```

---

## 🧮 Le Modèle SABR (Hagan et al., 2002)

Pour les deux portefeuilles, le modèle SABR suppose que le taux forward (taux LIBOR pour les caps, taux de swap pour les swaptions) $F$ et sa volatilité $\alpha$ suivent des processus stochastiques corrélés ($\rho$).
- **$\alpha$** : Volatilité court terme (niveau global du smile).
- **$\beta$** : Exposant CEV contrôlant le "backbone" (déplacement du smile face aux mouvements de taux).
- **$\rho$** : Corrélation entre le taux et sa vol (gère l'asymétrie/inclinaison des ailes).
- **$\nu$** : Vol de la vol (gère la courbure/le "smile" des ailes).

---

## 📌 1. Module Caps & Floors (`sabr_caps_floors/`)

Ce module calibre une surface de volatilité très dense (**120 expiries × 19 strikes**) issue de données réelles du marché USD du 13 juillet 2016.

* **Spécificités techniques :**
  * **Shifted SABR :** En environnement de taux bas (2016 : $F \approx 0.6\%$), un décalage standard du marché de 1% (100 bps) est introduit pour stabiliser la formule de Hagan et éviter les singularités à zéro.
  * **$\beta = 0$ Optimal :** L'analyse montre qu'un modèle Normal (Bachelier, $\beta=0$) s'ajuste beaucoup mieux aux données USD de 2016 qu'un modèle Log-normal ($\beta=1$), réduisant le RMSE global de près de 88%.
  * **Parité Cap-Floor :** Seule la surface des Caps est calibrée. Les prix des Floors en sont déduits sans arbitrage via la relation : `Floor(K) = Cap(K) - PV(Swap Payeur K)`.

---

## 📌 2. Module Swaptions (`sabr_swaptions/`)

Ce module gère le pricing des options sur taux de swap (Payer et Receiver Swaptions) sur une grille d'expiries et de tenors allant de 1 an à 30 ans.

* **Spécificités techniques :**
  * **Cotation en Spreads :** Les données de marché (`swaption_quotes.xlsx`) fournissent une nappe de volatilités ATM absolues et des grilles de spreads de strikes en points de base (-200 bps à +200 bps). Le script `data_loader.py` fusionne ces informations pour reconstruire les smiles absolus.
  * **Ancrage ATM (Pondération) :** L'algorithme des moindres carrés applique un poids 3x supérieur sur le point à la monnaie (spread = 0) pour forcer le modèle SABR à s'aligner parfaitement sur la volatilité de référence du marché.
  * **Pricing et Annuité (PV01) :** Contrairement aux caps, le pricing d'une swaption dépend de l'annuité du swap sous-jacent $A(T,\tau)$, calculée rigoureusement à partir de la courbe des taux.

---

## ⚙️ Dépendances & Installation

Les deux environnements utilisent les mêmes librairies standards de calcul quantitatif et de data science :

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## 🚀 Utilisation rapide

Chaque sous-dossier possède son propre Notebook d'analyse qui déroule le pipeline complet (Bootstrap $\rightarrow$ Data Loading $\rightarrow$ Calibration $\rightarrow$ Pricing $\rightarrow$ Plots).

### Exemple d'utilisation du Pricer de Swaptions :
```python
from pathlib import Path
from sabr_swaptions.data_loader import load_discount_factors
from sabr_swaptions.yield_curve import YieldCurve
from sabr_swaptions.pricer import swaption_price, black_greeks

# 1. Chargement de la courbe des taux spécifique aux swaptions
yc = YieldCurve(load_discount_factors("sabr_swaptions/data/market_discount_factors.xlsx"))

# 2. Extraction des paramètres du Swap sous-jacent (ex: Expiry 5y, Tenor 10y)
F = yc.swap_rate("5y", "10y")
annuity = yc.annuity("5y", "10y")
sigma_sabr = 0.172  # Volatilité obtenue via l'objet SABRSurface calibré

# 3. Pricing d'une Swaption Payer ATM (Notionnel 10M)
prix = swaption_price(F, K=F, sigma=sigma_sabr, T=5.0, annuity=annuity, notional=10_000_000, option_type="payer")
greeks = black_greeks(F, K=F, sigma=sigma_sabr, T=5.0, annuity=annuity, notional=10_000_000, option_type="payer")

print(f"Prix de la Swaption : {prix:,.2f} $")
print(f"Vega (pour 1% de vol) : {greeks['vega_1pct']:.2f} $")
```

---

## 📚 Référence
* Hagan, P., Kumar, D., Léger, A., Woodward, D. (2002). *Managing Smile Risk*. Wilmott Magazine.
