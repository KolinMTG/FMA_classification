## Dossier Data

- FMA_small : contient l'ensembles des fichier audio.mp3
- metadata : contient l'ensemble des data au format .csv
- tfrecords : contient l'ensemble des data après prétraitement sous forme de fichier .tfrecords
- models : contient tout les modèles tensorflow enregistré

### Meta data

#### filtered_tracks.csv
| track_id | genre |
| ---- | ---- |
| 19192 | Experimental |

#### Path Label
| path | label |
| ---- | ---- |
| data/FMA_small/124\124755.mp3 | 0 |

### Tfrecord : fichier tfrecord (natif tensorflow)

| Champ |	Type TFRecord / Format	| Contenu| 
| ----- | ----- | ----- | 
| spectrogram	| bytes_list	| Log-Mel spectrogram du segment, codé en float32, converti en bytes. Taille originale (H, W, 1) avant sérialisation.| 
| label	| int64_list	| Label entier correspondant au genre de l’audio.| 
| height	| int64_list	| Hauteur (H) du spectrogram avant sérialisation. |
|width	| int64_list	| Largeur (W) du spectrogram avant sérialisation.|

## Dossier src

- **cste.py** : contient l'ensemble des constantes pour le projet, si temps les mettre dans des classes séparé pour plus de clareté. 

- **logger.py** : contient un fichier de gestion/creation suppression des logs

- **others.py** : contient l'ensemble des fonction non utiles à la pipeline principale mais qui peuvent etre utiles lors de la construction du code, ou le teste de fonctionnalité. Contient notement un système de supressiond de fichier/dossier vers la corbeil .trash historisé.

- **data_utils.py** : Etape 00 du projet, permet de construire le CSV de l'ensemble des audio séléctionné et équilibré.

- **data_pretreat.py** : Etape 01 du projet, permet l'extraction et sauvegarde des fichier tfrecords fixé et utilisé pour l'entrainement.

- **data_split.py** : Etape 02 du proejt,  Gère les splits pour les data. Permet la création du fichier CSV 

- **model.py** : Contient l'ensemble des fonctionnalités realative la création des modèles. Les modèles sont enregistré dans le dossier model du dossier data. 

- **model_training.py** : Contient l'ensemble des fonctionnalités relatives à l'entrainement des modèle crées avec model via les données construites avec data_utils et pretrat_pipeline

- **model_evalution.py** : Contien l'ensemble des fonction permetant l'évaluation d'un modèle tf donnée

- **main_pipeline.py** : Contient l'ensemble des Etapes du projet pour former une pipeline complette utilisant les constantes du projet pour definir des data a utiliser, construire des modeles, les entrainer, les évaluer, séléctionner le meilleur et renvoyer ce meilleur model. Ainsi qu'une fonction permetant la prédiction d'un fichier audio donnée. 