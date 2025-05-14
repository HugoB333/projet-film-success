import streamlit as st
import pandas as pd
import joblib

# Message de démarrage
st.write("✅ Lancement réussi — app.py commence bien.")
st.title("🎬 Prédiction de succès d'un film")
st.write("📦 Étape 1 : chargement du modèle...")

# Chargement du modèle
try:
    model = joblib.load("model/model.pkl")
    st.write("✅ Modèle chargé avec succès.")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")

# Formulaire utilisateur
year = st.number_input("Année de sortie", min_value=1900, max_value=2100, value=2024)
runtime = st.slider("Durée (minutes)", min_value=30, max_value=300, value=120)
rating = st.slider("Note IMDb", min_value=0.0, max_value=10.0, step=0.1, value=7.0)
votes = st.number_input("Nombre de votes IMDb", min_value=0, value=10000)
revenue = st.number_input("Revenu (en millions $)", min_value=0.0, value=100.0)
metascore = st.slider("Metascore", min_value=0, max_value=100, value=70)

# Création du DataFrame utilisateur
input_data = {
    "Year": year,
    "Runtime (Minutes)": runtime,
    "Rating": rating,
    "Votes": votes,
    "Revenue (Millions)": revenue,
    "Metascore": metascore
}
df_input = pd.DataFrame([input_data])

# Recharger les colonnes utilisées lors de l'entraînement
expected_cols = joblib.load("model/columns.pkl")

# Ajouter les colonnes manquantes avec 0
for col in expected_cols:
    if col not in df_input.columns:
        df_input[col] = 0

# Réordonner les colonnes comme à l'entraînement
df_input = df_input[expected_cols]

# Prédiction
if st.button("Prédire"):
    try:
        prediction = model.predict(df_input)[0]
        if prediction == 1:
            st.success("✅ Ce film a de fortes chances d’être un succès !")
        else:
            st.error("❌ Ce film risque de ne pas rencontrer le succès.")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")