# 🔧 Nouveau `train_model.py` corrigé pour prédiction du revenu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Charger le dataset
df = pd.read_csv("data/Cleaned_csv.csv")

# Nettoyage de base
df = df.dropna()

# Liste des features explicatives pertinentes UNIQUEMENT
features = [
    "Genre", "Director", "Actors",
    "Year", "Runtime (Minutes)",
    "Rating", "Metascore"
]

# Définir X et y
X = df[features]
y = df["Revenue (Millions)"]

# Encodage des catégories en one-hot
X_encoded = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Sauvegarde du modèle et des colonnes utilisées
joblib.dump(model, "model/model.pkl")
joblib.dump(X_encoded.columns.tolist(), "model/columns.pkl")

print("✅ Modèle entraîné et sauvegardé pour la prédiction du revenu.")
