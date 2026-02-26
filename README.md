# 🏦 Scoring Churn — Détection de départ clients

> **Licence Professionnelle Data Mining**
> Mise en place d'une solution de détection de départ des clients
>
> 👥 Équipe : Nicolas · Marouane · Ilyes · Ephraim

---

## 📌 Contexte et objectifs

Une banque observe une **hausse des résiliations** de cartes de crédit. L'objectif de ce projet est de développer un **modèle prédictif probabiliste** capable d'anticiper les clients à risque de churn (attrition) avant qu'ils ne partent.

### Pourquoi ce projet ?
- 📉 Coût d'acquisition d'un nouveau client = 5× le coût de fidélisation
- 🎯 Cibler les bonnes actions de rétention nécessite d'identifier les clients à risque
- 📊 Dataset de **+10 000 clients** avec 19 caractéristiques (âge, revenu, comportement transactionnel...)

---

## 🏗️ Pipeline ML

```
Données brutes (CSV)
       │
       ▼
┌─────────────────────┐
│  1. Chargement      │  loader.py
│     & Renommage     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Nettoyage       │  cleaner.py
│  • Valeurs manq.    │  → Médiane / Moyenne / Mode
│  • Outliers IQR     │  → Clipping [Q1−1.5×IQR, Q3+1.5×IQR]
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Feature Eng.    │  engineering.py
│  • Discrétisation   │  → 4 classes par quartile
│  • Regroupement     │  → Modalités rares → "Autre"
│  • WOE / IV         │  → Pouvoir prédictif des variables
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Tests stats     │  stats_tests.py
│  • Mann-Whitney     │  → Variables quantitatives
│  • Chi² + V Cramer  │  → Variables qualitatives
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  5. Modélisation (SMOTE + Balanced) │  train.py
│  ┌────────────────┐ ┌─────────────┐ │
│  │ Régression     │ │   Forêt     │ │
│  │ Logistique     │ │  Aléatoire  │ │
│  │ AUC : 0.94     │ │ AUC : 0.96  │ │
│  └────────────────┘ └─────────────┘ │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Outputs         │
│  • Courbes ROC      │
│  • Matrices confus. │
│  • Import. variables│
│  • Modèles .pkl     │
└─────────────────────┘
```

---

## 📊 Résultats des modèles

| Métrique | Régression Logistique | Forêt Aléatoire |
|----------|----------------------|-----------------|
| **AUC-ROC** | **0.94** | **0.96** ✅ |
| Précision (résiliés) | 0.60 | 0.86 |
| Rappel (résiliés) | 0.87 | 0.70 |
| F1-score (résiliés) | 0.71 | 0.77 |
| Précision globale | 76% | 93% |

> 💡 **Recommandation** : La **Forêt Aléatoire** offre de meilleures performances globales (AUC 0.96, précision 93%). La Régression Logistique reste pertinente si l'on veut **maximiser le rappel** (détecter un maximum de résiliés, quitte à avoir plus de faux positifs).

---

## 🔑 Variables les plus prédictives

Les **3 variables clés** identifiées par les deux modèles :

| Rang | Variable | Importance |
|------|----------|------------|
| 🥇 | Nombre total de transactions | ★★★★★ |
| 🥈 | Montant total de transactions | ★★★★★ |
| 🥉 | Variation totale des transactions Q4/Q1 | ★★★★☆ |

---

## 📁 Structure du projet

```
scoring-churn/
│
├── main.py                    # ▶ Point d'entrée — lance le pipeline complet
├── requirements.txt           # Dépendances Python
├── README.md
├── .gitignore
│
├── src/                       # Code source
│   ├── data/
│   │   ├── loader.py          # Chargement & renommage des colonnes
│   │   └── cleaner.py         # Valeurs manquantes + outliers IQR
│   │
│   ├── features/
│   │   ├── engineering.py     # Discrétisation, regroupement, WOE/IV
│   │   └── stats_tests.py     # Mann-Whitney, Chi², V de Cramer
│   │
│   ├── models/
│   │   └── train.py           # Pipelines Régression Logistique + Random Forest
│   │
│   └── visualization/
│       └── plots.py           # Toutes les fonctions de visualisation
│
├── data/
│   ├── raw/                   # csv 
│   └── processed/             # Données transformées 
│
├── outputs/
│   ├── figures/               # Graphiques générés (ROC, confusion, etc.)

```

---

## 🛠️ Outils nécessaires pour lancer le projet

| Outil | Version utilisée |
|-------|-----------------|
| Python | 3.11.x |
| pip | 24.x |

Vérifiez vos versions :

```bash
python --version
pip --version
```

---

## 🚀 Installation et lancement

### 1. Cloner le projet

```bash
git clone https://github.com/marouane-nouara/scoring-churn.git
cd scoring-churn
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Placer le dataset

```bash
# Copier votre fichier BDD_PROJETS.csv dans data/raw/
cp /chemin/vers/BDD_PROJETS.csv data/raw/BDD_PROJETS.csv
```

### 5. Lancer le pipeline complet

```bash
python main.py --data data/raw/BDD_PROJETS.csv
```

Les résultats sont automatiquement sauvegardés dans :
- `outputs/figures/` — tous les graphiques (ROC, confusion, importances...)
- `outputs/models/` — modèles entraînés (`logistic_regression.pkl`, `random_forest.pkl`)

---

## 🧪 Lancer les tests

```bash
pytest tests/ -v
```

---

## 📦 Désactiver l'environnement virtuel

```bash
deactivate
```


## 🔬 Méthodologie détaillée

### Traitement des valeurs manquantes
- **4 variables** avec 49 valeurs manquantes → imputation par médiane/moyenne
- **8 variables** avec 7 valeurs manquantes → imputation par mode (catégorielles)

### Traitement des outliers (IQR Clipping)
```
IQR = Q3 − Q1
Borne inférieure = Q1 − 1.5 × IQR
Borne supérieure = Q3 + 1.5 × IQR
```
*Exemple : Limite de crédit réduite de 654M → max 23 828 €*

### Sélection des variables
Variables exclues après tests statistiques et analyse de corrélation :
- `Moyenne_disponible_pour_achats` (corrélée à 0.99 avec `Limite_de_crédit`)
- `Montant_total_transactions` (corrélée à 0.86 avec `Nombre_total_transactions`)
- `Ancienneté` (corrélée à 0.78 avec `Age`)

### Gestion du déséquilibre des classes
- Seulement **16% de clients résiliés** dans le dataset
- Technique : **SMOTE** (Synthetic Minority Oversampling Technique) + `class_weight='balanced'`
