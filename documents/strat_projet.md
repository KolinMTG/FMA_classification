# Strategie pour le projet 

## 1. Contraintes fondamentales à prendre en compte

Avant toute stratégie, il faut poser les contraintes réelles.
### Contraintes matérielles
- CPU limité (i7 11th gen pc portable)
- Pas de GPU local
- mémoire finie (16Go)
- temps de calcul non négligeable par entraînement

*Cela impose :*
- entraînement partiel
- élimination rapide
- pas de grid search exhaustif

### Contraintes données
- audio coûteux à traiter
- dataset volumineux

*Cela impose :*
- prétraitement unique et figé
- séparation stricte train / val / test

### Contraintes scientifiques
- éviter le surapprentissage
- éviter le data leakage
- métriques comparables entre expériences

## 2. Ce qui est réellement optimisable

Il est important de distinguer ce qui est structurel de ce qui est paramétrable.

### Non optimisable (à fixer une fois)
- type de représentation audio (log-Mel)
- durée des segments
- fréquence d’échantillonnage
- protocole de split


### Optimisable
- architecture CNN
- learning rate
- batch size
- régularisation
- scheduler
- profondeur du modèle


## 3. Différentes étapes du projet 

- extraction audio → une fois
- conversion TFRecord → une fois
- split train/val/test → une fois

Tout le reste doit être itératif.

## 4. Pipeline global prévue (vue d’ensemble)


### Etape 0 - Netoyage des données du dataset initial
Permet de selectionner les données : les audio à exploiter en priorité.

- sélection des classes
- vérification des fichiers audio existants
- équilibrage strict
- mapping label ↔ index
- génération CSV final

**Réalisé par la fonction : build_csv_pipeline_00**

### Étape 1 — Extraction des features pour les audio séléctionnés

- Génération TFRecords
- Le dataset ne bouge plus

**Réalisé par la fonction: build_tfrecord_from_dataframe_01**


### Étape 2 — Split définitif du dataset

**Objectif : garantir une évaluation propre.**
- split train / val / test
- test set jamais utilisé pendant l’optimisation
- stocker les index de split

**Réalisé par la fonction : data_split_pipeline (fichier data_split)**


### Étape 3 — Définition d’un espace de recherche

Avant d’entraîner quoi que ce soit, définir :
- bornes réalistes
- paramètres discrets / continus
- contraintes (ex: taille max du modèle)

**Voir fichier research_range.md**

### Étape 4 — Modèle de base (baseline)

Objectif : avoir un point de comparaison.

- CNN simple
- entraînement complet
- métriques de référence

**Réalisé par la fonction : A CODER**


### Étape 5 — Évaluation rapide (fitness proxy)


Chaque candidat :

- est entraîné partiellement
- sur train + val
- avec early stopping agressif
- sur peu d’epochs

La métrique n’est pas la performance finale, mais :
la capacité à apprendre rapidement sans diverger

**Réalisé par la fonction : A CODER**

### Étape 6 — Optimisation des hyperparamètres

Peu importe l’algorithme (GA, Hyperband, PBT), la logique est la même :

- générer des candidats
- évaluer rapidement
- éliminer les pires
- concentrer les ressources sur les meilleurs

**IMPORTANT : loguer toutes les expériences et versionner les config testés**
**Réalisé par la fonction : A CODER**

### Étape 7 — Sélection finale

Séléction d'une architecture stable pour le porjet
Ce choix doit être justifiable (pas juste la meilleure accuracy).

Rappel : Stabilité > performance brute
Un modèle légèrement moins performant mais stable est préférable.

**A mettre dans le rapport**

### Étape 8 — Entraînement final long

Seulement maintenant :

- dataset complet
- epochs élevés
- callbacks complets
- checkpoints
- monitoring fin

C’est le seul entraînement “cher”.

### Étape 9 — Évaluation finale sur le test set

Règle absolue :

le test set n’est utilisé qu’une seule fois
Tu produis :
- accuracy
- confusion matrix
- précision / rappel par classe


## 6. Stratégie optimale pour le projet

Compte tenu de ton contexte :

1. pipeline figée avec TFRecords
2. CNN simple mais robuste
3. optimisation sur entraînement partiel
4. élimination rapide
5. entraînement final unique
6. analyse qualitative des erreurs
