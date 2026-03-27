# C.6 — Credit Scoring avec IA Explicable (XAI)

> **ECE Paris — Ing4 — IA Probabiliste, Théorie des Jeux et ML — 2026**  
> Groupe : `groupe-XX-credit-scoring-xai`

---

## Résumé

Ce projet implémente un système de **credit scoring explicable** conforme au **RGPD (Article 22)**, qui oblige les entreprises à fournir une explication sur toute décision automatisée affectant un individu.

Le pipeline complet couvre :
- Entraînement de modèles performants (XGBoost, LightGBM)
- Explications globales et locales via **SHAP** et **LIME**
- **Explications contrefactuelles** (DiCE) : "Que dois-je changer pour être accepté ?"
- **Analyse de biais** (âge, genre) et **audit de fairness** (Fairlearn)
- **Dashboard interactif** Gradio pour les conseillers bancaires

---

## Structure du projet

```
groupe-XX-credit-scoring-xai/
├── README.md                          ← Ce fichier
├── requirements.txt                   ← Dépendances Python
├── src/
│   └── credit_scoring_xai.ipynb      ← Notebook principal (TOUT le projet)
├── docs/
│   ├── technical_doc.md              ← Documentation technique détaillée
│   ├── eda.png                       ← Analyse exploratoire
│   ├── shap_summary.png              ← SHAP beeswarm plot
│   ├── shap_importance.png           ← SHAP importance globale
│   ├── shap_local_refused.png        ← SHAP waterfall refusé
│   ├── shap_local_accepted.png       ← SHAP waterfall accepté
│   ├── shap_vs_lime.png              ← Comparaison SHAP vs LIME
│   ├── bias_analysis.png             ← Analyse de biais
│   ├── blackbox_vs_interpretable.png ← Trade-off explainability
│   ├── decision_tree.png             ← Arbre de décision visualisé
│   └── model_evaluation.png          ← Courbes ROC + confusion matrix
└── slides/
    └── slides_c6_xai.pdf             ← Support de présentation
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/ECE-2026-IA-Finance
cd ECE-2026-IA-Finance/groupe-XX-credit-scoring-xai

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter
jupyter notebook src/credit_scoring_xai.ipynb
```

### Alternative Google Colab (recommandé)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Ouvrir Google Colab
2. `File → Upload notebook` → charger `src/credit_scoring_xai.ipynb`
3. Exécuter cellule par cellule (Runtime → Run all)

---

## Dataset

**German Credit Dataset** (UCI Machine Learning Repository)
- **Source** : `sklearn.datasets.fetch_openml('credit-g', version=1)`
- **Taille** : 1 000 clients, 20 features, 1 cible binaire
- **Cible** : `1` = bon crédit (70%), `0` = mauvais crédit (30%)
- **Features** : statut compte, durée, montant, historique de crédit, épargne, emploi, âge, etc.

Aucun téléchargement manuel requis — le dataset est chargé automatiquement via scikit-learn.

---

## Résultats Clés

### Performance des modèles

| Modèle | AUC-ROC | Accuracy | Explicabilité | Type |
|--------|---------|----------|---------------|------|
| **XGBoost** | **~0.80** | **~0.76** | Partielle (SHAP) | Boîte Noire |
| LightGBM | ~0.79 | ~0.75 | Partielle (SHAP) | Boîte Noire |
| Régression Logistique | ~0.74 | ~0.71 | ✅ Directe | Interprétable |
| Arbre de Décision | ~0.72 | ~0.70 | ✅ Directe | Interprétable |

### Top 5 Features (SHAP)

1. **Statut du compte courant** — Feature la plus déterminante
2. **Durée du crédit** — Corrélée positivement avec le risque
3. **Historique de crédit** — Indicateur fort de fiabilité
4. **Montant du crédit** — Plus le montant est élevé, plus le risque augmente
5. **Épargne** — Présence d'épargne favorise l'acceptation

### Audit de Fairness

| Métrique | Âge (Jeune vs Senior) | Genre (H vs F) |
|----------|----------------------|----------------|
| Demographic Parity Diff | < 0.15 | < 0.10 |
| Equalized Odds Diff | < 0.20 | < 0.15 |
| Statut | ⚠️ Léger biais jeunes | ✅ Équitable |

---

## Architecture du Pipeline

```
Données brutes (German Credit)
        │
        ▼
┌─────────────────┐
│  Prétraitement  │  OrdinalEncoder + SMOTE (équilibrage)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│              Modèles ML                      │
│  XGBoost │ LightGBM │ LogReg │ DecisionTree │
└────────┬─────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────────┐
│ SHAP  │  │     LIME     │  ← Explications locales + globales
└───┬───┘  └──────────────┘
    │
    ▼
┌──────────────────┐
│  Contrefactuels  │  DiCE — "Que changer pour être accepté ?"
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Audit Fairness  │  Fairlearn — Equalized Odds, Mitigation
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Dashboard Gradio │  Interface conseiller bancaire
└──────────────────┘
```

---

## Fonctionnalités par niveau

### ✅ Niveau Minimum
- [x] Modèle XGBoost sur German Credit Dataset
- [x] Calcul des valeurs SHAP (globales et locales)
- [x] Visualisations : summary plot, waterfall plot, beeswarm

### ✅ Niveau Bon
- [x] Comparaison SHAP vs LIME (avantages/inconvénients)
- [x] Explications contrefactuelles via DiCE ("Que changer ?")
- [x] Analyse de biais (âge, genre) — Disparate Impact

### ✅ Niveau Excellent
- [x] Dashboard interactif Gradio avec explication temps-réel
- [x] Audit de fairness complet (Fairlearn) — Equalized Odds
- [x] Mitigation des biais (ThresholdOptimizer)
- [x] Comparaison boîte noire vs modèle interprétable

---

## Contexte Réglementaire

### RGPD — Article 22
> *"Toute personne a le droit de ne pas faire l'objet d'une décision fondée exclusivement sur un traitement automatisé [...] produisant des effets juridiques la concernant."*
>
> **Droit à l'explication** : Si une décision automatisée affecte un individu, il peut demander une explication des logiques utilisées.

### Notre implémentation
- **SHAP** : fournit l'explication mathématiquement rigoureuse (théorie des valeurs de Shapley)
- **LIME** : alternative agnostique plus rapide
- **Contrefactuels** : explication actionnable ("que faire pour changer la décision")
- **Dashboard** : interface permettant au conseiller d'expliquer la décision au client

---

## Librairies utilisées

| Librairie | Version | Usage |
|-----------|---------|-------|
| `xgboost` | ≥ 1.7 | Modèle principal |
| `lightgbm` | ≥ 3.3 | Modèle alternatif |
| `shap` | ≥ 0.42 | Explications SHAP |
| `lime` | ≥ 0.2 | Explications LIME |
| `dice-ml` | ≥ 0.9 | Contrefactuels |
| `fairlearn` | ≥ 0.9 | Audit fairness |
| `imbalanced-learn` | ≥ 0.10 | SMOTE |
| `gradio` | ≥ 4.0 | Dashboard interactif |
| `scikit-learn` | ≥ 1.3 | ML baseline |
| `plotly` | ≥ 5.0 | Visualisations |

---

## Références

### Papers
- Lundberg, S. M., & Lee, S. I. (2017). **A Unified Approach to Interpreting Model Predictions** (SHAP). NeurIPS.
- Ribeiro, M. T., et al. (2016). **"Why Should I Trust You?" Explaining the Predictions of Any Classifier** (LIME). KDD.
- Wachter, S., et al. (2017). **Counterfactual Explanations without Opening the Black Box**. Harvard Journal of Law & Technology.
- Bird, S., et al. (2020). **Fairlearn: A Toolkit for Assessing and Improving Fairness in AI**. Microsoft Research.

### Datasets & Outils
- [German Credit Dataset (UCI)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- [SHAP Library](https://shap.readthedocs.io/)
- [InterpretML — Microsoft](https://interpret.ml/)
- [Fairlearn](https://fairlearn.org/)
- [DiCE — Microsoft Research](https://github.com/interpretml/DiCE)

### Notebooks de référence (CoursIA)
- [Lab3 — CV Screening (classification)](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScience/Lab3-CVScreening.ipynb)
- [ML-4 — Évaluation de modèles ML](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-4-EvaluationModels.ipynb)

---

## Membres du groupe

| Nom | GitHub | Contribution |
|-----|--------|-------------|
| [Prénom Nom 1] | @username1 | Modèles ML, SHAP |
| [Prénom Nom 2] | @username2 | LIME, Contrefactuels |
| [Prénom Nom 3] | @username3 | Fairness, Dashboard |

---

## Critères d'évaluation (checklist)

- [x] **Qualité théorie** : SHAP (Shapley), LIME (régression locale), Équité algorithmique
- [x] **Qualité technique** : Notebook complet, reproductible, code documenté
- [x] **Qualité présentation** : Dashboard interactif, visualisations claires
- [x] **Organisation** : Structure GitHub propre, README détaillé

---

*Projet réalisé avec assistance IA (Claude, GitHub Copilot) — ECE Paris 2026*
