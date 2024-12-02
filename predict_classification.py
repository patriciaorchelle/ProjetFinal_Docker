import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Charger le modèle et les données de validation
def load_model_and_data(export_dir="exports"):
    model_path = os.path.join(export_dir, "best_model.pkl")
    val_data_path = os.path.join(export_dir, "validation_data.csv")

    if not os.path.exists(model_path) or not os.path.exists(val_data_path):
        raise FileNotFoundError("Le modèle ou les données de validation n'ont pas été trouvés.")

    # Charger le modèle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Charger les données de validation
    val_data = pd.read_csv(val_data_path)
    X_val = val_data.drop("target", axis=1)
    y_val = val_data["target"]

    return model, X_val, y_val

# Faire des prédictions et évaluer les performances
def predict_and_evaluate(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Précision sur les données de validation : {accuracy:.4f}")
    print("\nRapport de classification :")
    print(classification_report(y_val, y_val_pred))

# Fonction principale
def main():
    # Charger le modèle et les données
    model, X_val, y_val = load_model_and_data()

    # Faire des prédictions et évaluer les performances
    predict_and_evaluate(model, X_val, y_val)

if __name__ == "__main__":
    main()
