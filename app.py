import streamlit as st
import pandas as pd
import joblib

# Titre
st.title("🎬 Estimation du revenu d’un film (en millions $)")

# Chargement du modèle
try:
    model = joblib.load("model/model.pkl")
    expected_cols = joblib.load("model/columns.pkl")
    st.success("✅ Modèle chargé avec succès.")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")

# Charger les données pour remplir les menus déroulants
df_raw = pd.read_csv("data/Cleaned_csv.csv")
genres = df_raw["Genre"].dropna().unique().tolist()
directors = df_raw["Director"].dropna().unique().tolist()
actors = df_raw["Actors"].dropna().unique().tolist()

# Inputs utilisateur
genre = st.selectbox("Genre", genres)
director = st.selectbox("Réalisateur", directors)
actors_choice = st.selectbox("Acteurs principaux", actors)
year = st.number_input("Année de sortie", min_value=1900, max_value=2100, value=2024)
runtime = st.slider("Durée (minutes)", min_value=30, max_value=300, value=120)
rating = st.slider("Note IMDb", min_value=0.0, max_value=10.0, step=0.1, value=7.0)
metascore = st.slider("Metascore", min_value=0, max_value=100, value=70)
success = st.selectbox("Succès du film (1 = oui, 0 = non)", [1, 0])

# Création du DataFrame utilisateur
input_data = {
    "Genre": genre,
    "Director": director,
    "Actors": actors_choice,
    "Year": year,
    "Runtime (Minutes)": runtime,
    "Rating": rating,
    "Metascore": metascore,
    "Success": success
}
df_input = pd.DataFrame([input_data])
df_input = pd.get_dummies(df_input)

# Compléter les colonnes manquantes
for col in expected_cols:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[expected_cols]

# Prédiction
if st.button("Prédire le revenu estimé"):
    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Revenu estimé : {prediction:.2f} millions de dollars")
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
