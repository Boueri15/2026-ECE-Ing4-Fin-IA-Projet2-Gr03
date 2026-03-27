# Documentation Technique — C.6 Credit Scoring XAI

## 1. Fondements Théoriques

### 1.1 SHAP — Shapley Additive exPlanations

**Origine** : Théorie des jeux coopératifs (Lloyd Shapley, 1953 — Prix Nobel 2012).

La valeur de Shapley d'un joueur `i` dans un jeu coopératif est définie comme :

```
φᵢ = Σ [|S|! (n-|S|-1)! / n!] × [v(S∪{i}) - v(S)]
     S⊆N\{i}
```

Où :
- `N` = ensemble de toutes les features
- `S` = coalition de features
- `v(S)` = valeur (prédiction) de la coalition S
- `n` = nombre total de features

**Interprétation** : La valeur SHAP d'une feature = sa contribution marginale moyenne à la prédiction, sur toutes les coalitions possibles.

**Propriétés garanties** :
1. **Efficience** : La somme des SHAP values = prédiction − valeur de base
2. **Symétrie** : Deux features identiques ont les mêmes SHAP values
3. **Nullité** : Feature sans impact → SHAP = 0
4. **Additivité** : SHAP d'un modèle combiné = somme des SHAP

**TreeSHAP** (implémentation pour arbres) : O(TLD²) au lieu de O(TL·2ⁿ) — algorithme exact et efficace pour XGBoost/LightGBM.

### 1.2 LIME — Local Interpretable Model-agnostic Explanations

**Principe** : Autour de chaque instance `x`, LIME :
1. Génère des perturbations `z'` dans l'espace des features
2. Obtient les prédictions `f(z)` pour ces perturbations
3. Pèse les perturbations par leur proximité à `x` : `πₓ(z) = exp(-D(x,z)²/σ²)`
4. Ajuste un modèle linéaire interprétable `g` :

```
ξ(x) = argmin_{g∈G} L(f, g, πₓ) + Ω(g)
```

**Limite principale** : Pas de garantie de stabilité — deux runs peuvent donner des explications différentes.

### 1.3 Explications Contrefactuelles (DiCE)

**Définition** : Un contrefactuel `x'` est le point le plus proche de `x` dans l'espace des features tel que `f(x') ≠ f(x)`.

**Formalisation DiCE** :

```
argmin_{x'} distance(x, x') + λ × diverse_penalty
subject to: f(x') = desired_class
            x'_immutable = x_immutable  (features non-modifiables)
```

**Application pratique** :
- Features immuables : âge, genre → jamais modifiées
- Features actionnables : durée, montant, épargne → peuvent être changées

### 1.4 Fairness — Métriques

**Demographic Parity (Parité Démographique)** :
```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
```
Tous les groupes ont le même taux d'acceptation.

**Equalized Odds** :
```
P(Ŷ=1 | Y=y, A=0) = P(Ŷ=1 | Y=y, A=1)  pour y ∈ {0,1}
```
Même TPR et même FPR pour tous les groupes.

**Disparate Impact Ratio** (règle 4/5 EEOC) :
```
DIR = P(Ŷ=1 | A=minorité) / P(Ŷ=1 | A=majorité)
```
- DIR < 0.8 → Discrimination légale (règle des 4/5)
- DIR ∈ [0.8, 1.25] → Zone acceptable
- DIR > 1.25 → Discrimination inverse

---

## 2. Choix Techniques

### 2.1 Pourquoi XGBoost comme modèle principal ?

| Critère | Justification |
|---------|--------------|
| Performance | AUC ~0.80 sur German Credit, SOTA pour données tabulaires |
| Compatibilité SHAP | TreeSHAP = algorithme exact en O(TLD²) |
| Régularisation | L1/L2 intégrée → moins de surapprentissage |
| Robustesse | Gère bien les données tabulaires mixtes |

### 2.2 Pourquoi SMOTE ?

German Credit est déséquilibré (70/30). SMOTE génère des instances synthétiques de la classe minoritaire en interpolant dans l'espace des features.

**Alternative testée** : class_weight='balanced' dans XGBoost → moins efficace sur ce dataset.

### 2.3 OrdinalEncoder vs OneHotEncoder

On utilise `OrdinalEncoder` (et non `OneHotEncoder`) car :
- XGBoost gère nativement les features catégorielles encodées ordinalement
- Évite l'explosion dimensionnelle (20 features × N catégories)
- **Important pour SHAP** : SHAP TreeExplainer fonctionne mieux avec moins de features

### 2.4 ThresholdOptimizer (Fairlearn)

Au lieu de ré-entraîner le modèle, on optimise le seuil de décision par groupe pour satisfaire les contraintes de fairness.

Avantage : Rapide, ne nécessite pas de ré-entraînement. Inconvénient : Peut réduire l'accuracy globale.

---

## 3. Guide d'interprétation des résultats

### 3.1 Lire un SHAP Waterfall Plot

```
Base value (prédiction moyenne)
│
├── Feature A: +0.15  → A augmente la probabilité d'acceptation
├── Feature B: -0.23  → B diminue la probabilité d'acceptation
├── Feature C: +0.08  → ...
│
└── Prédiction finale = base + Σ SHAP_i
```

### 3.2 Interpréter le Disparate Impact

Exemple : DIR(jeunes) = 0.72
- Signification : Les jeunes ont 28% moins de chances d'être acceptés que la moyenne
- Légalement : 0.72 < 0.8 → Violation potentielle de la règle des 4/5 (EEOC)
- Action recommandée : Audit approfondi + mitigation

### 3.3 SHAP vs LIME — Quand utiliser quoi ?

| Situation | Recommandation |
|-----------|---------------|
| Expliquer pour audit réglementaire | **SHAP** (propriétés garanties) |
| Expliquer rapidement pour client | **LIME** (plus simple, plus rapide) |
| Comparer plusieurs clients | **SHAP global** (summary plot) |
| "Que changer ?" | **DiCE** (contrefactuels) |
| Modèle simple requis | **Coefficients LR** ou **Règles DT** |

---

## 4. Reproducibilité

```python
SEED = 42  # Seed globale pour reproductibilité

# Toutes les fonctions stochastiques utilisent SEED:
train_test_split(..., random_state=SEED)
SMOTE(random_state=SEED)
XGBClassifier(random_state=SEED)
LGBMClassifier(random_state=SEED)
LogisticRegression(random_state=SEED)
DecisionTreeClassifier(random_state=SEED)
```

**Version Python recommandée** : Python 3.10+ (testée sur 3.10 et 3.11)

---

## 5. Extensions Possibles

1. **Conformalized Conformal Prediction** : Ajouter des intervalles de confiance aux prédictions
2. **Monotonic Constraints** : Contraindre XGBoost à respecter des monotonies métier (plus de crédit = plus de risque)
3. **SHAP Interactions** : Analyser les interactions entre features avec SHAP interaction values
4. **Online Learning** : Mise à jour incrémentale du modèle sur nouveaux clients
5. **LLM Explanations** : Utiliser GPT/Claude pour générer des explications en langage naturel à partir des SHAP values

---

## 6. Références Bibliographiques

1. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
2. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*.
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" *KDD 2016*.
4. Mothilal, R. K., et al. (2020). Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations. *ACM FAccT 2020*.
5. Hardt, M., et al. (2016). Equality of Opportunity in Supervised Learning. *NeurIPS 2016*.
6. Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press.
7. European Commission. (2021). *Proposal for Artificial Intelligence Act (AIA)*. COM(2021) 206 final.
