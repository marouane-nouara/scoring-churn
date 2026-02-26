"""
plots.py
────────
Fonctions de visualisation regroupées et réutilisables.
Chaque fonction accepte un paramètre `save_dir` optionnel
pour sauvegarder les figures dans outputs/figures/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix


# ── Utilitaire ────────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, filename: str | None, save_dir: Path | None) -> None:
    """Sauvegarde ou affiche la figure selon les paramètres."""
    if save_dir and filename:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / filename, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ── Valeurs manquantes ────────────────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame, title: str = "Valeurs manquantes", save_dir=None):
    """Affiche un heatmap des valeurs manquantes par colonne."""
    import missingno as msno
    fig, ax = plt.subplots(figsize=(14, 6))
    msno.matrix(df, ax=ax, sparkline=False)
    ax.set_title(title, fontsize=14)
    _save_or_show(fig, "missing_values.png", save_dir)


# ── Analyse univariée ─────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, target: str = "Statut_d_attrition", save_dir=None):
    """Distribution de la variable cible (barplot + pourcentages)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts  = df[target].value_counts()
    labels  = ["Actif (0)", "Résilié (1)"]
    bars    = ax.bar(labels, counts.values, color=["#2196F3", "#F44336"], alpha=0.85)

    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_title("Distribution du statut d'attrition", fontsize=13)
    ax.set_ylabel("Nombre de clients")
    fig.tight_layout()
    _save_or_show(fig, "target_distribution.png", save_dir)


def plot_boxplots(df: pd.DataFrame, variables: list[str], save_dir=None):
    """Boxplots côte à côte pour un ensemble de variables numériques."""
    n = len(variables)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        if var in df.columns:
            sns.boxplot(x=df[var], ax=axes[i], color="#90CAF9")
            axes[i].set_title(var, fontsize=10)

    # Masquer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boxplots des variables numériques", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_or_show(fig, "boxplots.png", save_dir)


# ── Corrélations ──────────────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, variables: list[str], save_dir=None):
    """Heatmap de la matrice de corrélation des variables quantitatives."""
    corr = df[variables].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="YlGnBu", linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Matrice de corrélation", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, "correlation_matrix.png", save_dir)


# ── Modèles ───────────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_proba, model_name: str, color: str = "#1565C0", save_dir=None):
    """Courbe ROC avec score AUC."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard")
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title(f"Courbe ROC — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, f"roc_{model_name.lower().replace(' ', '_')}.png", save_dir)
    return roc_auc


def plot_confusion_matrix(y_test, y_pred, model_name: str, cmap: str = "Blues", save_dir=None):
    """Matrice de confusion annotée."""
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Actif", "Résilié"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    ax.set_title(f"Matrice de confusion — {model_name}")
    fig.tight_layout()
    _save_or_show(fig, f"cm_{model_name.lower().replace(' ', '_')}.png", save_dir)


def plot_feature_importance_logreg(
    pipeline,
    variables_categoricielles: list[str],
    X: pd.DataFrame,
    save_dir=None,
):
    """Importance des variables pour la régression logistique (somme |coefficients|)."""
    ohe              = pipeline.named_steps["preprocessing"].named_transformers_["cat"]
    ohe_feat_names   = ohe.get_feature_names_out(variables_categoricielles)
    other_cols       = X.select_dtypes(exclude="object").columns.tolist()
    all_feat_names   = np.concatenate([ohe_feat_names, other_cols])
    coefficients     = pipeline.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({"feature": all_feat_names, "coefficient": coefficients})

    feat_to_var = {}
    for var in variables_categoricielles:
        for feat in ohe_feat_names:
            if feat.startswith(var + "_"):
                feat_to_var[feat] = var

    coef_df["variable"] = coef_df["feature"].map(feat_to_var).fillna(coef_df["feature"])
    imp_df = (
        coef_df.groupby("variable")["coefficient"]
        .apply(lambda x: np.sum(np.abs(x)))
        .reset_index()
        .sort_values("coefficient", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["variable"], imp_df["coefficient"], color="#42A5F5")
    ax.set_xlabel("Importance (Σ |coefficients|)")
    ax.set_title("Importance des variables — Régression Logistique")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "importance_logreg.png", save_dir)


def plot_feature_importance_rf(
    pipeline,
    variables_categoricielles: list[str],
    X: pd.DataFrame,
    save_dir=None,
):
    """Importance des variables pour la forêt aléatoire."""
    ohe              = pipeline.named_steps["preprocessing"].named_transformers_["cat"]
    ohe_feat_names   = ohe.get_feature_names_out(variables_categoricielles)
    other_cols       = X.select_dtypes(exclude="object").columns.tolist()
    all_feat_names   = np.concatenate([ohe_feat_names, other_cols])
    importances      = pipeline.named_steps["model"].feature_importances_

    imp_df = pd.DataFrame({"feature": all_feat_names, "importance": importances})

    feat_to_var = {
        feat: var
        for var in variables_categoricielles
        for feat in ohe_feat_names
        if feat.startswith(var + "_")
    }

    imp_df["variable"] = imp_df["feature"].map(feat_to_var).fillna(imp_df["feature"])
    grouped = (
        imp_df.groupby("variable")["importance"]
        .sum()
        .reset_index()
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(grouped["variable"], grouped["importance"], color="#66BB6A")
    ax.set_xlabel("Importance")
    ax.set_title("Importance des variables — Forêt Aléatoire")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "importance_rf.png", save_dir)
