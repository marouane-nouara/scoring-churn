"""
stats_tests.py
──────────────
Tests statistiques pour la sélection de variables :
  - Mann-Whitney (variables quantitatives vs cible binaire)
  - Chi² + V de Cramer (variables qualitatives vs cible binaire)
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency


def test_mann_whitney(
    df: pd.DataFrame,
    variables: list[str],
    target: str = "Statut_d_attrition",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Teste si chaque variable quantitative diffère significativement
    entre clients actifs (0) et résiliés (1).

    H0 : Les distributions des deux groupes sont identiques.
    H1 : Il existe une différence significative entre les deux groupes.

    Args:
        df        : DataFrame contenant les données
        variables : Liste des variables quantitatives à tester
        target    : Variable cible binaire
        alpha     : Seuil de signification (défaut : 0.05)

    Returns:
        DataFrame avec [variable, statistique, p_valeur, significatif]
    """
    results = []
    group_0 = df[df[target] == 0]
    group_1 = df[df[target] == 1]

    for var in variables:
        if var not in df.columns:
            continue
        stat, p = mannwhitneyu(group_0[var], group_1[var], alternative="two-sided")
        results.append({
            "variable":     var,
            "statistique":  round(stat, 2),
            "p_valeur":     round(p, 6),
            "significatif": "✅ Oui" if p < alpha else "❌ Non",
        })

    return pd.DataFrame(results).sort_values("p_valeur")


def test_chi2_cramer(
    df: pd.DataFrame,
    variables: list[str],
    target: str = "Statut_d_attrition",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Teste l'association entre chaque variable qualitative et la variable cible
    via le test du Chi² et mesure la force via le V de Cramer.

    V de Cramer :
      < 0.1  : aucune/très faible association
      0.1–0.3: faible
      0.3–0.5: modérée
      > 0.5  : forte

    Args:
        df        : DataFrame contenant les données
        variables : Liste des variables qualitatives à tester
        target    : Variable cible
        alpha     : Seuil de signification (défaut : 0.05)

    Returns:
        DataFrame avec [variable, chi2, p_valeur, v_cramer, association, significatif]
    """
    results = []

    for var in variables:
        if var not in df.columns:
            continue

        contingency = pd.crosstab(df[target], df[var])
        chi2, p, dof, _ = chi2_contingency(contingency)

        n       = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        v       = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        if v < 0.1:
            force = "Très faible"
        elif v < 0.3:
            force = "Faible"
        elif v < 0.5:
            force = "Modérée"
        else:
            force = "Forte"

        results.append({
            "variable":     var,
            "chi2":         round(chi2, 4),
            "p_valeur":     round(p, 6),
            "v_cramer":     round(v, 4),
            "association":  force,
            "significatif": "✅ Oui" if p < alpha else "❌ Non",
        })

    return pd.DataFrame(results).sort_values("v_cramer", ascending=False)
