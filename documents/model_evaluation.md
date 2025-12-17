

## 2.2 Métriques globales
| Nom	| Var| 	Rôle| 
| ---- | ---- | ---- | 
| Accuracy| 	acc| 	Performance brute| 
| Macro F1| 	f1_macro| 	Équité inter-classes| 
| Weighted F1| 	f1_weighted| Robustesse| 
| Loss validation| 	loss_val| 	Stabilité| 
| Overfitting gap	| gap| 	Généralisation| 
	​

## 2.3 Métriques dynamiques (essentielles pour GA)

Ces métriques permettent d’évaluer un modèle partiellement entraîné.

| Nom | 	Description | 
| ---- | ---- | 
| epochs_trained | 	Nombre d’epochs effectifs | 
| best_epoch	 | Epoch du meilleur val | 
| convergence_speed | 	Epoch où val_loss minimale | 
| divergence_flag	 | Explosion loss / NaN | 
| learning_slope	 | Dérivée loss sur premiers epochs | 

## Fitness Value
Definition d'un alpha, beta, gamma, sigma :

**fiteness = alpha⋅f1macro​−beta⋅gap−gamma⋅lossval​−sigma⋅complexity**

-Favorise la stabilité 
-Pénalise la complexité, et le sur-apprentissage

## Stockage des données dans un CSV avec ces informations précises 
model_id,generation,fitness,f1_macro,accuracy,loss_val,loss_train,gap,epochs_trained,num_params,train_time_sec,status

| Nom du champ     | Type informatique | Description                                                                                                                       |
| ---------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `model_id`       | `str`             | Identifiant unique du modèle évalué. Permet de relier les résultats à une architecture, une configuration ou un génome précis.    |
| `generation`     | `int`             | Numéro de génération dans l’algorithme d’optimisation (ex: algorithme génétique). Permet d’analyser la progression dans le temps. |
| `fitness`        | `float`           | Score global utilisé pour la sélection des modèles. Combinaison pondérée de métriques (F1, overfitting, complexité, etc.).        |
| `f1_macro`       | `float`           | F1-score macro-averaged sur le jeu de validation. Mesure l’équilibre des performances entre les classes.                          |
| `accuracy`       | `float`           | Accuracy globale sur le jeu de validation. Indique la proportion de prédictions correctes.                                        |
| `loss_val`       | `float`           | Valeur minimale de la fonction de perte sur le jeu de validation. Indicateur de généralisation.                                   |
| `loss_train`     | `float`           | Valeur minimale de la fonction de perte sur le jeu d’entraînement. Sert à détecter l’overfitting.                                 |
| `gap`            | `float`           | Différence `loss_val - loss_train`. Mesure directe du surapprentissage du modèle.                                                 |
| `epochs_trained` | `int`             | Nombre réel d’epochs effectués avant arrêt (early stopping inclus). Indicateur de vitesse de convergence.                         |
| `num_params`     | `int`             | Nombre total de paramètres entraînables du modèle. Sert à pénaliser les modèles trop complexes.                                   |
| `train_time_sec` | `float`           | Temps total d’entraînement en secondes pour ce modèle. Permet d’intégrer le coût de calcul.                                       |
| `status`         | `str`             | État final de l’évaluation (`OK`, `FAILED`, `DIVERGED`). Permet de filtrer les modèles invalides.                                 |

