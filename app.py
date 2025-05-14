import streamlit as st
import pandas as pd
import joblib

# 🎬 Titre principal
st.title("🎬 Estimation du revenu d’un film (en millions $)")

# Chargement du modèle
try:
    model = joblib.load("model/model.pkl")
    expected_cols = joblib.load("model/columns.pkl")
    st.success("✅ Modèle chargé avec succès.")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")

# 🎞️ Liste fixe des genres (comme dans l'exemple que tu as donné)
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
          'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller',
          'War', 'Western']

# 🎯 Extraction des choix depuis le CSV
df_raw = pd.read_csv("data/Cleaned_csv.csv")
directors = sorted(df_raw["Director"].dropna().unique().tolist())
actors = sorted(df_raw["Actors"].dropna().unique().tolist())

# 🎛️ Inputs utilisateur
genre = st.multiselect("🎞️ Genres", genres, default=['Action', 'Adventure'])
director = st.selectbox("🎬 Réalisateur", directors)
actors_choice = st.multiselect("👥 Acteurs principaux", actors, default=actors[:1])
year = st.number_input("📅 Année de sortie", min_value=1900, max_value=2100, value=2024)
runtime = st.slider("⏱️ Durée (minutes)", min_value=30, max_value=300, value=120)
rating = st.slider("⭐ Note IMDb", min_value=0.0, max_value=10.0, step=0.1, value=7.0)
metascore = st.slider("🧠 Metascore", min_value=0, max_value=100, value=70)

# 💾 Préparation des données
genre_str = ",".join(genre)
actors_str = ", ".join(actors_choice)

input_data = {
    "Genre": genre_str,
    "Director": director,
    "Actors": actors_str,
    "Year": year,
    "Runtime (Minutes)": runtime,
    "Rating": rating,
    "Metascore": metascore
}
df_input = pd.DataFrame([input_data])
df_input = pd.get_dummies(df_input)

# 🧩 Compléter avec colonnes manquantes
for col in expected_cols:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[expected_cols]

# 📊 Affichage des données utilisateur
st.subheader("📋 Données saisies")
st.dataframe(df_input)

# 🎯 Prédiction
if st.button("📈 Prédire le revenu estimé"):
    try:
        prediction = model.predict(df_input)[0]
        st.success(f"💰 Revenu estimé : **{prediction:.2f} millions de dollars**")
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
