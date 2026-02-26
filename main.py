"""
main.py
───────
Point d'entrée principal du projet de scoring churn.

Lance le pipeline complet :
  1. Chargement des données
  2. Nettoyage (missing values + outliers)
  3. Feature engineering (discrétisation + regroupement)
  4. Tests statistiques (Mann-Whitney + Chi² / V de Cramer)
  5. Modélisation (Régression Logistique + Forêt Aléatoire)
  6. Évaluation et sauvegarde des résultats

Usage :
    python main.py --data data/raw/BDD_PROJETS.csv
"""

import argparse
from pathlib import Path

from src.data.loader import load_data
from src.data.cleaner import impute_missing_values, clip_outliers_iqr
from src.features.engineering import (
    discretize_quantitative,
    regroup_rare_modalities,
    compute_woe_iv,
    VARIABLES_QUANTITATIVES,
)
from src.features.stats_tests import test_mann_whitney, test_chi2_cramer
from src.models.train import (
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
    split_data,
    save_model,
    VARIABLES_MODELE,
    TARGET,
)
from src.visualization.plots import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance_logreg,
    plot_feature_importance_rf,
)

# ── Dossiers de sortie ────────────────────────────────────────────────────────
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR  = Path("outputs/models")


def main(data_path: str):
    print("\n" + "=" * 55)
    print("  SCORING CHURN — Pipeline complet")
    print("=" * 55 + "\n")

    # ── 1. Chargement ──────────────────────────────────────────
    df = load_data(data_path)

    # ── 2. Nettoyage ───────────────────────────────────────────
    df_clean = impute_missing_values(df)
    df_clean = clip_outliers_iqr(df_clean)

    # ── 3. Feature Engineering ─────────────────────────────────
    df_disc, bornes = discretize_quantitative(df_clean)
    df_feat         = regroup_rare_modalities(df_disc)

    # ── 4. Tests statistiques ──────────────────────────────────
    print("\n📊 Mann-Whitney (variables quantitatives) :")
    mw_results = test_mann_whitney(df_feat, VARIABLES_QUANTITATIVES)
    print(mw_results.to_string(index=False))

    variables_qualitatives = [
        "Genre", "Niveau_éducation2", "Statut_marital2",
        "Catégorie_de_revenu_annuel2", "Catégorie_de_carte",
        "Age_classe", "Nombre_personnes_charge_classe",
        "Ancienneté_classe", "Nombre_total_de_relations_classe",
        "Mois_inactifs_12_derniers_mois_classe",
        "Nombre_de_contacts_12_derniers_mois_classe",
        "Limite_de_crédit_classe",
        "Variation_totale_montant_Q4_Q1_classe",
        "Montant_total_transactions_classe",
        "Nombre_total_transactions_classe",
        "Variation_totale_transactions_Q4_Q1_classe",
        "Taux_moyen_d_utilisation_classe",
    ]

    print("\n📊 Chi² + V de Cramer (variables qualitatives) :")
    chi2_results = test_chi2_cramer(df_feat, variables_qualitatives)
    print(chi2_results.to_string(index=False))

    print("\n📊 WOE / Information Value :")
    iv_results = compute_woe_iv(df_feat, variables_qualitatives)
    print(iv_results.to_string(index=False))

    # ── Visualisations exploratoires ───────────────────────────
    plot_target_distribution(df_feat, save_dir=FIGURES_DIR)

    variables_num = [v for v in VARIABLES_QUANTITATIVES if v in df_feat.columns]
    plot_correlation_matrix(df_feat, [TARGET] + variables_num, save_dir=FIGURES_DIR)

    # ── 5. Préparation du jeu de données modèle ────────────────
    df_modele  = df_feat[VARIABLES_MODELE].copy()
    X_train, X_test, y_train, y_test = split_data(df_modele)

    print(f"\n📦 Jeu d'entraînement : {X_train.shape} | Test : {X_test.shape}")
    print(f"   Distribution cible (train) :\n{y_train.value_counts(normalize=True).round(3)}\n")

    # ── 6. Modélisation ────────────────────────────────────────
    print("🔧 Entraînement — Régression Logistique...")
    pipeline_lr = train_logistic_regression(X_train, y_train)
    res_lr      = evaluate_model(pipeline_lr, X_test, y_test, "Régression Logistique")

    print("🔧 Entraînement — Forêt Aléatoire...")
    pipeline_rf = train_random_forest(X_train, y_train)
    res_rf      = evaluate_model(pipeline_rf, X_test, y_test, "Forêt Aléatoire")

    # ── 7. Visualisations modèles ──────────────────────────────
    plot_roc_curve(y_test, res_lr["y_proba"], "Régression Logistique", color="#1565C0", save_dir=FIGURES_DIR)
    plot_roc_curve(y_test, res_rf["y_proba"], "Forêt Aléatoire",       color="#2E7D32", save_dir=FIGURES_DIR)

    plot_confusion_matrix(y_test, res_lr["y_pred"], "Régression Logistique", cmap="Blues",  save_dir=FIGURES_DIR)
    plot_confusion_matrix(y_test, res_rf["y_pred"], "Forêt Aléatoire",       cmap="Greens", save_dir=FIGURES_DIR)

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    plot_feature_importance_logreg(pipeline_lr, cat_cols, X_train, save_dir=FIGURES_DIR)
    plot_feature_importance_rf(pipeline_rf,     cat_cols, X_train, save_dir=FIGURES_DIR)

    # ── 8. Sauvegarde des modèles ──────────────────────────────
    save_model(pipeline_lr, MODELS_DIR / "logistic_regression.pkl")
    save_model(pipeline_rf, MODELS_DIR / "random_forest.pkl")

    # ── Résumé final ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RÉSULTATS FINAUX")
    print("=" * 55)
    print(f"  Régression Logistique — AUC : {res_lr['auc']:.4f}")
    print(f"  Forêt Aléatoire        — AUC : {res_rf['auc']:.4f}")
    winner = "Forêt Aléatoire" if res_rf["auc"] > res_lr["auc"] else "Régression Logistique"
    print(f"\n  🏆 Meilleur modèle : {winner}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring Churn — Pipeline ML")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/BDD_PROJETS.csv",
        help="Chemin vers le fichier CSV source",
    )
    args = parser.parse_args()
    main(args.data)
