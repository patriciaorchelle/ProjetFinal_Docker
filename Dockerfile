# Utiliser une image de base Python 3.12
FROM python:3.12-alpine

# Installer les dépendances système nécessaires pour scikit-learn
RUN apk update && \
    apk add --no-cache \
    build-base \
    libatlas-dev \
    gfortran \
    && rm -rf /var/cache/apk/*
    
# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers locaux dans le conteneur
COPY . /app


# Mettre à jour pip et installer les dépendances nécessaires
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exécuter les deux étapes : 1) Entraînement, 2) Prédiction
CMD python train_classifier.py && python predict_classification.py
