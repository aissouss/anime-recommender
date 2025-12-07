# app.py

import streamlit as st
from src.recommender import TfidfRecommender, EmbeddingRecommender

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
)

st.title("🎌 Système de recommandation d'animes")
st.markdown(
    """
Ce projet propose des recommandations d'animes basées sur leur contenu.

- **TF-IDF** : modèle classique basé sur les genres / texte
- **BERT embeddings** : modèle plus avancé utilisant des représentations sémantiques
"""
)

@st.cache_resource
def load_tfidf_model():
    return TfidfRecommender("data/anime_clean.csv")

@st.cache_resource
def load_embedding_model():
    return EmbeddingRecommender("data/anime_clean.csv")


# Sélection du modèle
model_type = st.radio(
    "Choisis le modèle :",
    ["TF-IDF (rapide)", "BERT embeddings (plus avancé)"],
    horizontal=True,
)

# Chargement du modèle
with st.spinner("Chargement du modèle..."):
    if model_type.startswith("TF-IDF"):
        model = load_tfidf_model()
        current_model_name = "TF-IDF"
    else:
        model = load_embedding_model()
        current_model_name = "BERT"

# Sélection du type
types = model.available_types()
col1, col2 = st.columns([2, 1])
with col1:
    title = st.text_input("Entre le nom d'un anime que tu aimes :")
with col2:
    selected_type = st.selectbox("Filtrer par type :", types)

top_n = st.slider("Nombre de recommandations :", min_value=3, max_value=15, value=5)

if st.button("🔍 Recommander") and title:
    with st.spinner("Recherche des recommandations..."):
        output = model.recommend(title, type_filter=selected_type, top_n=top_n)

    if "error" in output:
        st.warning(output["error"])
        if output["suggestions"]:
            st.info("Suggestions possibles : " + ", ".join(output["suggestions"]))
    else:
        st.success(f"Voici les recommandations ({current_model_name}) :")

        results = output["results"]

        for item in results:
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border-radius: 10px;
                        padding: 12px;
                        margin-bottom: 8px;
                        border: 1px solid #444;
                        background-color: #11111110;
                    ">
                        <h4 style="margin-bottom:4px;">{item['name']}</h4>
                        <p style="margin:0;">
                            <b>Type :</b> {item['type']} &nbsp; | 
                            <b>Note moyenne :</b> {item['rating']:.2f} / 10 &nbsp; | 
                            <b>Score de similarité :</b> {item['score']:.3f}
                        </p>
                        <p style="margin-top:4px;">
                            <b>Genres :</b> {item['genre']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

