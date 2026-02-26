"""
loader.py
─────────
Chargement et renommage du dataset BDD_PROJETS.csv
"""

import pandas as pd
from pathlib import Path

# ── Mapping des noms de colonnes anglais → français ──────────────────────────
COLUMN_NAMES = {
    "CLIENTNUM":                "Numéro_de_client",
    "Attrition_Flag":           "Statut_d_attrition",
    "Customer_Age":             "Age",
    "Gender":                   "Genre",
    "Dependent_count":          "Nombre_personnes_charge",
    "Education_Level":          "Niveau_éducation",
    "Marital_Status":           "Statut_marital",
    "Income_Category":          "Catégorie_de_revenu_annuel",
    "Card_Category":            "Catégorie_de_carte",
    "Months_on_book":           "Ancienneté",
    "Total_Relationship_Count": "Nombre_total_de_relations",
    "Months_Inactive_12_mon":   "Mois_inactifs_12_derniers_mois",
    "Contacts_Count_12_mon":    "Nombre_de_contacts_12_derniers_mois",
    "Credit_Limit":             "Limite_de_crédit",
    "Total_Revolving_Bal":      "Solde_total_renouvelable",
    "Avg_Open_To_Buy":          "Moyenne_disponible_pour_achats",
    "Total_Amt_Chng_Q4_Q1":     "Variation_totale_montant_Q4_Q1",
    "Total_Trans_Amt":          "Montant_total_transactions",
    "Total_Trans_Ct":           "Nombre_total_transactions",
    "Total_Ct_Chng_Q4_Q1":      "Variation_totale_transactions_Q4_Q1",
    "Avg_Utilization_Ratio":    "Taux_moyen_d_utilisation",
}


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Charge le CSV, renomme les colonnes et encode la variable cible.

    Args:
        filepath: Chemin vers BDD_PROJETS.csv

    Returns:
        DataFrame prêt pour le nettoyage.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8", sep=";")
    df.rename(columns=COLUMN_NAMES, inplace=True)

    # Encodage de la variable cible : 0 = client actif, 1 = client résilié
    df["Statut_d_attrition"] = df["Statut_d_attrition"].replace(
        {"Existing Customer": 0, "Attrited Customer": 1}
    )

    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df
