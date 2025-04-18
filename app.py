import streamlit as st
from api_utils import search_anime, search_game

st.set_page_config(page_title="Recommandation Animes & Jeux")

st.title("🎌 Système de Recommandation")
st.markdown("Recommande des **animes** ou des **jeux vidéo** en se basant sur les titres recherchés.")

media_type = st.radio("Type :", ["Anime", "Jeu vidéo"])
query = st.text_input("Quel titre aimes-tu ?")

api_key = ""
if media_type == "Jeu vidéo":
    api_key = st.text_input("Clé API RAWG (gratuite)", type="password")

if st.button("Rechercher") and query:
    if media_type == "Anime":
        results = search_anime(query)
    else:
        if not api_key:
            st.warning("Entre ta clé API RAWG.io")
            st.stop()
        results = search_game(query, api_key)

    if results:
        st.success("Voici les résultats :")
        for item in results:
            st.write("🎯", item)
    else:
        st.warning("Aucun résultat trouvé.")


