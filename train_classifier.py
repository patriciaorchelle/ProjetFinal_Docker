import os
import pickle
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Générer des données synthétiques
def generate_data():
    X, y = make_classification(
        n_samples=15000,
        n_features=20,
        n_informative=2,
        n_redundant=10,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=42
    )
    # Convertir en DataFrame pour plus de flexibilité
    data = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    data["target"] = y
    #plt.figure(figsize=(12, 8))
    #sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    #plt.title('Matrice de Corrélation')
    #plt.show()
    colonnes_a_conserver = ['feature_1', 'feature_2','feature_5','feature_10','feature_11','feature_12','feature_13','feature_14','feature_15','feature_16','feature_18','target']
    data = data[colonnes_a_conserver]

    return data

# Séparer les données en ensembles d'entraînement, test et validation
def split_data(data):
    X = data.drop("target", axis=1)
    y = data["target"]
    
    # Séparer les données en train et un ensemble temporaire
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Diviser l'ensemble temporaire en test et validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
    )
    return X_train, X_test, X_val, y_train, y_test, y_val

# Entraîner plusieurs modèles et choisir le meilleur
def train_and_select_best_model(X_train, y_train, X_test, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    best_model = None
    best_accuracy = 0
    best_model_name = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        print(f"{model_name} - Précision sur les données de test : {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print(f"\nMeilleur modèle : {best_model_name} avec une précision de {best_accuracy:.4f}")
    return best_model, best_model_name

# Exporter le meilleur modèle et les données de validation
def export_model_and_data(model, X_val, y_val, export_dir="exports"):
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # Sauvegarder le modèle
    model_path = os.path.join(export_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Sauvegarder les données de validation
    val_data = pd.concat([X_val, y_val], axis=1)
    val_data_path = os.path.join(export_dir, "validation_data.csv")
    val_data.to_csv(val_data_path, index=False)

    print(f"Modèle exporté dans : {model_path}")
    print(f"Données de validation exportées dans : {val_data_path}")

# Fonction principale
def main():
    # Étape 1 : Générer les données
    data = generate_data()

    # Étape 2 : Séparer les données
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(data)

    # Étape 3 : Entraîner plusieurs modèles et choisir le meilleur
    best_model, best_model_name = train_and_select_best_model(X_train, y_train, X_test, y_test)

    # Étape 4 : Exporter le meilleur modèle et les données de validation
    export_model_and_data(best_model, X_val, y_val)

if __name__ == "__main__":
    main()
