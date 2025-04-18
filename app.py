import streamlit as st
from recommender import AnimeRecommender

reco = AnimeRecommender("anime.csv")

st.set_page_config(page_title="Recommandateur d'animes")
st.title("🎌 Système de Recommandation d’Animes")
st.markdown("Ce prototype recommande des animes similaires à celui que tu entres, en se basant sur les genres.")

titre = st.text_input("Entre un anime que tu aimes :", placeholder="Ex : Naruto")

if st.button("Recommander") and titre:
    result = reco.recommend(titre)
    if isinstance(result, str):
        st.warning(result)
    else:
        st.success("Voici les recommandations :")
        for i, anime in enumerate(result, start=1):
            st.write(f"**{i}.** {anime}")
