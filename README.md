# Music Genre Classification from Audio Signals

## Free Music Archive (FMA) – Medium Dataset
https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium
- Projet : Music Genre Classification
- Type : Audio signal processing & Machine Learning
- Dataset : Free Music Archive (FMA) – Medium
- Langage principal : Python
- Date : 2025
- Cadre : Projet académique
 
## Objectif du projet

L’objectif de ce projet est de concevoir et d’analyser une chaîne complète de **classification automatique de genres musicaux** à partir de signaux audio bruts.
Le projet se concentre sur la maîtrise du **pipeline complet**, depuis le chargement des fichiers audio jusqu’à l’évaluation des performances du modèle, sans utiliser de caractéristiques pré-calculées.

Le dataset utilisé est FMA Medium (Free Music Archive), choisi pour sa qualité méthodologique, la présence de métadonnées riches (genres, artistes, splits officiels) et l’absence des problèmes majeurs connus dans des datasets plus anciens comme GTZAN.

Le dataset est accessible via Kaggle à l’adresse suivante :
https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium
Dataset de Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson.
International Society for Music Information Retrieval Conference (ISMIR), 2017.


## 2. Étapes à réaliser (TODO)

Les étapes principales du projet sont les suivantes :

#### Analyse du dataset

-Compréhension de la structure des fichiers audio

-Exploration des métadonnées (genres, artistes, splits)

-Sélection d’un niveau de genre adapté à la classification

#### Prétraitement audio

-Chargement des fichiers audio
-Normalisation du format (fréquence d’échantillonnage, mono/stéréo)
-Gestion de la durée des pistes
-Découpage éventuel en segments temporels

#### Représentation du signal

-Transformation du signal audio en représentations exploitables
-Choix justifié des descripteurs audio ou des représentations temps-fréquence

#### Construction du dataset d’apprentissage

-Association audio ↔ métadonnées
-Respect strict des splits fournis par FMA
-Prévention des fuites de données (data leakage)

#### Entraînement des modèles

- Mise en place de modèles de référence (baselines)
- Entraînement de modèles plus avancés si pertinent
- Réglage des hyperparamètres

#### Évaluation et analyse

- Mesures de performance globales
- Analyse des confusions entre genres
- Étude des erreurs et des limites du modèle

#### Discussion et conclusion

- Interprétation des résultats
- Limites du dataset et du modèle
- Perspectives d’amélioration
- Rédaction rapport et présentation


## 3. Environnement de travail

Le projet est réalisé en Python, dans un environnement virtuel *Conda* dédié, afin de garantir la reproductibilité et l’isolation des dépendances. Version Python : 3.10 pour assurer compatibilité avec Tensorflow et sk-learn

#### L’environnement comprend notamment :

- Les bibliothèques standards de calcul scientifique
- Les outils de traitement du signal audio
- Les bibliothèques de machine learning et deep learning
- Des outils de visualisation et d’analyse

#### Pour installer l'environement sur votre ordinateur :

Utiliser les commandes suivantes : 

Création de l'environement virtuel Conda et activation
```python
 conda create -n FMA python=3.10
 conda activate FMA
```
Installations des library/dépenances
```python
pip install -r requirements.txt
```

Verifirication de l'installation
```python
conda list
```


#### 4. Auteur(s)

- Colin Manyri : colin.manyri@etu.utc.fr
