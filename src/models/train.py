"""
train.py
────────
Entraînement des modèles de scoring :
  - Régression Logistique (avec SMOTE + class_weight='balanced')
  - Forêt Aléatoire       (avec SMOTE + class_weight='balanced')

Les deux modèles sont entraînés via sklearn Pipeline + imblearn SMOTE
pour gérer le déséquilibre des classes (~16% de churn).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ── Variables retenues après sélection (Chi², IV, corrélation) ───────────────
VARIABLES_MODELE = [
    "Age_classe",
    "Nombre_personnes_charge_classe",
    "Nombre_total_de_relations_classe",
    "Mois_inactifs_12_derniers_mois_classe",
    "Nombre_de_contacts_12_derniers_mois_classe",
    "Limite_de_crédit_classe",
    "Variation_totale_montant_Q4_Q1_classe",
    "Montant_total_transactions_classe",
    "Nombre_total_transactions_classe",
    "Variation_totale_transactions_Q4_Q1_classe",
    "Taux_moyen_d_utilisation_classe",
    "Catégorie_de_carte",
    "Statut_d_attrition",  # variable cible
]

TARGET = "Statut_d_attrition"
TEST_SIZE = 0.30
RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le préprocesseur OneHot pour les colonnes catégorielles."""
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)],
        remainder="passthrough",
    )


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ImbPipeline:
    """
    Entraîne une régression logistique avec :
      - SMOTE pour rééquilibrer les classes
      - class_weight='balanced' pour donner plus de poids aux résiliés

    Returns:
        Pipeline entraîné
    """
    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    pipeline = ImbPipeline([
        ("preprocessing", preprocessor),
        ("smote",         SMOTE(random_state=RANDOM_STATE)),
        ("model",         model),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ImbPipeline:
    """
    Entraîne une forêt aléatoire avec :
      - SMOTE pour rééquilibrer les classes
      - class_weight='balanced' pour donner plus de poids aux résiliés

    Returns:
        Pipeline entraîné
    """
    preprocessor = build_preprocessor(X_train)
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    pipeline = ImbPipeline([
        ("preprocessing", preprocessor),
        ("smote",         SMOTE(random_state=RANDOM_STATE)),
        ("model",         model),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """
    Évalue un pipeline entraîné et affiche les métriques clés.

    Returns:
        dict avec AUC, rapport de classification
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Actif (0)", "Résilié (1)"]))
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"{'='*50}\n")

    return {"model_name": model_name, "auc": auc, "y_pred": y_pred, "y_proba": y_proba}


def split_data(df_modele: pd.DataFrame) -> tuple:
    """Sépare les features de la cible et effectue le train/test split."""
    X = df_modele.drop(TARGET, axis=1)
    y = df_modele[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def save_model(pipeline, path: str | Path) -> None:
    """Sauvegarde un pipeline entraîné avec joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"💾 Modèle sauvegardé : {path}")


def load_model(path: str | Path):
    """Charge un pipeline sauvegardé."""
    return joblib.load(path)
