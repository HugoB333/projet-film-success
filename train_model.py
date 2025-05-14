import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Charger les données
df = pd.read_csv("data/Cleaned_csv.csv")

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# Définir les features à garder
features = [
    "Genre", "Director", "Actors", "Year",
    "Runtime (Minutes)", "Rating", "Metascore", "Success"
]

X = df[features]
y = df["Revenue (Millions)"]

# Encodage des variables catégorielles
X = pd.get_dummies(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entraîner le modèle
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Sauvegarder le modèle et les colonnes
joblib.dump(model, "model/model.pkl")
joblib.dump(X.columns.tolist(), "model/columns.pkl")

print("✅ Modèle entraîné avec les bonnes features et sauvegardé.")
