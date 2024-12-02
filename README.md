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
 
Entrainement des modèles
Prédition
