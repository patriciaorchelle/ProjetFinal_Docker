# Explication du processus de fonctionnement
# Objectif

  • Simuler un projet d'entreprise réel où plusieurs développeurs travaillent sur le même
    projet et chacun développe une fonctionnalité dans une branche diérente puis
    fusionne sa version nale avec la branche principale (main).
  • Créer ensuite un docker qui se base sur le repository Github (branche main) en
    rajoutant le Dockerle.
  • Rajouter un docker-compose pour faciliter la mise le paramétrage et la mise à jour du docker.

  
# 1. Génération de donnes
Cette partie est consacrée à la génération des données en se basant sur la laibrairie `make_classification`de sklearn.
En faisant varier les paramètres de cette fonction, on est arrivé à déduire que la meilleure combinaison des paramètres est :
  - n_samples=20000
  - n_informative=5
  - n_classes=2
  - n_clusters_per_class=3
  - n_features=50
  - random_state=42
  L'analyse et la sélection des colonnes bien corrélation entre les variables et les prédicteurs et la variable cible.
 
# 2. Entrainement des modèles
Une fois les meilleurs paramètres de générations obtenus, les meilleures colonnes déterminées, nous avons entraîné plusieurs modèles,précisement:
  - `LogisticRegression`
  - `SVM`
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
  - `LogisticRegression`
  - `KNN`

Une première boucle `for`nous a permit de sélectionner trois modèles qui performent mieux:
  - `SVM`
  - `RandomForestClassifier`
  -  `KNN`

Nous avons ensuite utilisé ces modèles dans le fichier EDA_datageneration.ipynb pour aller vite.
  - Le méthode Elbow nous a permis prendre une plage restreinte des valeurs de nombre de voisions à prendre ( dans un intervalle [1,20], on a seulement pris 3,5 et 7 pour le gridsearch)
  - L'utilisation du gridsearch nous a permis de trouver la meilleure combinaison des hyperparamètres de chaque modèle.

Lorsque la meilleure combinaison est trouvée, nous avons créé le fichier .py qui reprend :
    a. Génération de données : Le code génère un jeu de données synthétiques avec 15 000 échantillons et 20 caractéristiques à l'aide de `make_classification` de `scikit-learn`. Certaines caractéristiques sont informatives, d'autres redondantes. Une           sélection de colonnes est ensuite effectuée pour réduire le nombre de caractéristiques utilisées.
    
    b. Séparation des données: Le jeu de données est divisé en ensembles d'entraînement, de test et de validation à l'aide de `train_test_split`. L'argument `stratify=y` est utilisé pour maintenir la distribution des classes dans les sous-ensembles.
    
    c. Entraînement et sélection du meilleur modèle : Trois modèles (Random Forest, SVM et Gradient Boosting) sont entraînés sur les données d'entraînement. La précision de chaque modèle est calculée sur l'ensemble de test. Le modèle ayant la       
         meilleure précision est sélectionné.
    
    d. Exportation du modèle et des données : Le meilleur modèle est enregistré dans un fichier binaire à l'aide de `pickle`. Les données de validation sont également sauvegardées dans un fichier CSV pour une utilisation ultérieure.
    
    # Points clés :
    - Modularité : Le code est bien structuré, avec des fonctions claires et séparées pour chaque étape du processus.
    - Utilisation de `stratify` : Cette approche garantit que la distribution des classes est maintenue dans les sous-ensembles de données, ce qui est important pour les problèmes de classification.
    - Approche simple et efficace : Le code utilise une méthode simple mais efficace pour entraîner et comparer plusieurs modèles.
    
