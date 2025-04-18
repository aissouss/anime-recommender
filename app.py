import streamlit as st
from recommender import ContentRecommender

# Charger le modèle
model = ContentRecommender("data.csv")

# Titre et description
st.set_page_config(page_title="Système de recommandation IA")
st.title("🎌 Système de recommandation d’animes et de jeux vidéo")
st.markdown("Ce système utilise un modèle basé sur le contenu avec IA (TF-IDF + cosine similarity).")

# Entrée utilisateur
media_type = st.radio("Type :", ["anime", "game"])
title = st.text_input("Entre le nom d’un anime ou d’un jeu que tu aimes :")

if st.button("Recommander") and title:
    results = model.recommend(title, media_type)
    
    if isinstance(results, str):
        st.warning(results)
    else:
        st.success("Voici les recommandations :")
        for i, item in enumerate(results, start=1):
            st.write(f"**{i}.** {item}")


