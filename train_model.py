import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Charger le CSV
df = pd.read_csv("data/Cleaned_csv.csv")

# Afficher les colonnes (pour vérification, tu peux enlever après)
print("Colonnes disponibles :", df.columns)

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# Définir X et y
X = df.drop("Success", axis=1)
y = df["Success"]

# Encodage des variables catégorielles automatiquement
X = pd.get_dummies(X)

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarde
joblib.dump(model, "model/model.pkl")

print("✅ Modèle entraîné et sauvegardé avec succès.")