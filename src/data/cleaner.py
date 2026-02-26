"""
cleaner.py
──────────
Nettoyage du dataset :
  - Traitement des valeurs manquantes (médiane / moyenne / mode)
  - Détection et correction des valeurs aberrantes par méthode IQR
"""

import pandas as pd
import numpy as np


# ── Valeurs manquantes ────────────────────────────────────────────────────────

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation des valeurs manquantes :
      - Colonnes numériques  → médiane pour 'Limite_de_crédit', moyenne pour les autres
      - Colonnes catégorielles → mode (valeur la plus fréquente)

    Args:
        df: DataFrame brut avec potentiellement des NaN

    Returns:
        DataFrame sans valeurs manquantes
    """
    df_clean = df.copy()

    # Limite de crédit : imputation par la médiane (moins sensible aux extremes)
    if "Limite_de_crédit" in df_clean.columns:
        df_clean["Limite_de_crédit"].fillna(df_clean["Limite_de_crédit"].median(), inplace=True)

    # Autres colonnes numériques → moyenne
    num_cols = df_clean.select_dtypes(include=["number"]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

    # Colonnes catégorielles → mode
    for col in df_clean.select_dtypes(include=["object"]).columns:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)

    print(f"✅ Imputation terminée. Valeurs manquantes restantes : {df_clean.isnull().sum().sum()}")
    return df_clean


# ── Valeurs aberrantes (IQR) ──────────────────────────────────────────────────

def clip_outliers_iqr(df: pd.DataFrame, target_col: str = "Statut_d_attrition") -> pd.DataFrame:
    """
    Détecte et corrige les valeurs aberrantes via la méthode IQR (Interquartile Range).
    Utilise le clipping pour ramener les valeurs extrêmes dans les bornes acceptables.

    Formule :
        IQR  = Q3 − Q1
        LB   = Q1 − 1.5 × IQR   (borne inférieure)
        UB   = Q3 + 1.5 × IQR   (borne supérieure)

    La variable cible est exclue du traitement.

    Args:
        df:         DataFrame après imputation
        target_col: Nom de la variable cible à exclure

    Returns:
        DataFrame avec outliers corrigés par clipping
    """
    df_out = df.copy()
    num_cols = [c for c in df_out.select_dtypes(include=["number"]).columns if c != target_col]

    outlier_report = {}
    for col in num_cols:
        Q1  = df_out[col].quantile(0.25)
        Q3  = df_out[col].quantile(0.75)
        iqr = Q3 - Q1
        lb  = Q1 - 1.5 * iqr
        ub  = Q3 + 1.5 * iqr

        n_outliers = ((df_out[col] < lb) | (df_out[col] > ub)).sum()
        outlier_report[col] = n_outliers

        df_out[col] = df_out[col].clip(lower=lb, upper=ub)

    total_outliers = sum(outlier_report.values())
    print(f"✅ Outliers corrigés (IQR clipping) — {total_outliers} valeurs modifiées au total")
    return df_out
