"""
engineering.py
──────────────
Transformation des variables :
  - Discrétisation par quartiles
  - Regroupement des modalités peu fréquentes
  - Calcul WOE / IV
"""

import pandas as pd
import numpy as np


# ── Variables numériques à discrétiser ───────────────────────────────────────
VARIABLES_QUANTITATIVES = [
    "Age", "Nombre_personnes_charge", "Ancienneté", "Nombre_total_de_relations",
    "Mois_inactifs_12_derniers_mois", "Nombre_de_contacts_12_derniers_mois",
    "Limite_de_crédit", "Solde_total_renouvelable", "Moyenne_disponible_pour_achats",
    "Variation_totale_montant_Q4_Q1", "Montant_total_transactions",
    "Nombre_total_transactions", "Variation_totale_transactions_Q4_Q1",
    "Taux_moyen_d_utilisation",
]


def _assign_class(value: float, quartiles: pd.Series) -> str:
    """Assigne une classe quartile à une valeur."""
    if value <= quartiles[0.25]:
        return "Classe 1"
    elif value <= quartiles[0.5]:
        return "Classe 2"
    elif value <= quartiles[0.75]:
        return "Classe 3"
    return "Classe 4"


def discretize_quantitative(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Discrétise les variables quantitatives en 4 classes selon leurs quartiles.

    Returns:
        df_disc   : DataFrame enrichi avec colonnes '_classe'
        bornes    : dictionnaire des bornes par variable
    """
    df_disc = df.copy()
    bornes  = {}

    for var in VARIABLES_QUANTITATIVES:
        if var not in df_disc.columns:
            continue
        q = df_disc[var].quantile([0, 0.25, 0.5, 0.75, 1])
        bornes[var] = q.values
        df_disc[f"{var}_classe"] = df_disc[var].apply(lambda x: _assign_class(x, q))

    print(f"✅ Discrétisation terminée — {len(bornes)} variables transformées en classes")
    return df_disc, bornes


def regroup_rare_modalities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe les modalités rares en 'Autre' pour réduire le bruit.

    Règles appliquées :
      - Niveau_éducation   : College, Graduate, High School, Uneducated → reste ; autres → 'Autre'
      - Statut_marital     : Married → reste ; autres → 'Autre'
      - Catégorie_de_revenu_annuel : $40K-$60K, $60K-$80K, $80K-$120K → reste ; autres → 'Autre'
    """
    df_out = df.copy()

    modalites_education = {"College", "Graduate", "High School", "Uneducated"}
    modalites_marital   = {"Married"}
    modalites_revenu    = {"$40K - $60K", "$60K - $80K", "$80K - $120K"}

    df_out["Niveau_éducation2"] = df_out["Niveau_éducation"].apply(
        lambda x: x if x in modalites_education else "Autre"
    )
    df_out["Statut_marital2"] = df_out["Statut_marital"].apply(
        lambda x: x if x in modalites_marital else "Autre"
    )
    df_out["Catégorie_de_revenu_annuel2"] = df_out["Catégorie_de_revenu_annuel"].apply(
        lambda x: x if x in modalites_revenu else "Autre"
    )

    print("✅ Regroupement des modalités rares terminé")
    return df_out


# ── WOE / IV ─────────────────────────────────────────────────────────────────

def compute_woe_iv(
    df: pd.DataFrame,
    variables: list[str],
    target: str = "Statut_d_attrition",
) -> pd.DataFrame:
    """
    Calcule le WOE (Weight of Evidence) et l'IV (Information Value)
    pour une liste de variables qualitatives.

    WOE  = ln(% Non EVENT / % EVENT)
    IV   = Σ (% Non EVENT − % EVENT) × WOE

    Interprétation de l'IV :
      < 0.02  : peu ou pas de pouvoir prédictif
      0.02–0.1: faible
      0.1–0.3 : modéré
      > 0.3   : fort

    Returns:
        DataFrame récapitulatif avec colonnes [variable, IV_total]
    """
    results = []

    for var in variables:
        if var not in df.columns:
            continue

        table = df.pivot_table(index=var, columns=target, aggfunc="size", fill_value=0)
        table.columns = ["Non EVENT", "EVENT"]

        total_event     = table["EVENT"].sum()
        total_non_event = table["Non EVENT"].sum()

        table["% EVENT"]     = table["EVENT"] / total_event
        table["% Non EVENT"] = table["Non EVENT"] / total_non_event

        # Éviter log(0)
        table.replace({"% EVENT": {0: 1e-4}, "% Non EVENT": {0: 1e-4}}, inplace=True)

        table["WOE"] = np.log(table["% Non EVENT"] / table["% EVENT"])
        table["IV"]  = (table["% Non EVENT"] - table["% EVENT"]) * table["WOE"]
        table.replace([np.inf, -np.inf], np.nan, inplace=True)

        total_iv = table["IV"].sum()
        results.append({"variable": var, "IV_total": round(total_iv, 4)})

    iv_df = pd.DataFrame(results).sort_values("IV_total", ascending=False)
    return iv_df
