# app.py
import streamlit as st

from src.recommender import TfidfRecommender, EmbeddingRecommender

DATA_PATH = "data/anime_clean.csv"

st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")

st.title("🎌 Système de recommandation d'animes")
st.caption(
    "Recommandation basée contenu : TF-IDF (rapide) ou embeddings de phrases (sémantique)."
)


@st.cache_resource(show_spinner=False)
def load_model(kind: str):
    """Le modèle est construit une seule fois, puis réutilisé entre les reruns."""
    if kind == "TF-IDF":
        return TfidfRecommender(DATA_PATH)
    return EmbeddingRecommender(DATA_PATH)


kind = st.radio(
    "Modèle :",
    ["TF-IDF", "Embeddings (BERT)"],
    horizontal=True,
    help="TF-IDF compare les mots ; les embeddings comparent le sens.",
)
kind = "TF-IDF" if kind == "TF-IDF" else "BERT"

try:
    with st.spinner(f"Chargement du modèle {kind}…"):
        model = load_model(kind)
except ImportError as exc:
    # Dépendance manquante : on le dit clairement au lieu de planter.
    st.error(f"Modèle indisponible : {exc}")
    st.stop()
except FileNotFoundError:
    st.error(f"Fichier introuvable : {DATA_PATH}. Lance d'abord `python prepare_data.py`.")
    st.stop()

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    title = st.text_input("Un anime que tu aimes :", placeholder="Cowboy Bebop")
with col2:
    selected_type = st.selectbox("Type :", model.available_types())
with col3:
    top_n = st.number_input("Résultats :", min_value=3, max_value=20, value=5)

search = st.button("🔍 Recommander", type="primary")

# On stocke la sortie en session : sinon le moindre changement de widget
# relance le script et efface les résultats affichés.
if search and title.strip():
    st.session_state["output"] = model.recommend(title, selected_type, int(top_n))
elif search:
    st.warning("Entre d'abord le nom d'un anime.")

output = st.session_state.get("output")

if output and "error" in output:
    st.warning(output["error"])
    if output["suggestions"]:
        st.caption("Tu voulais peut-être dire :")
        for i, suggestion in enumerate(output["suggestions"]):
            if st.button(suggestion, key=f"sugg_{i}"):
                st.session_state["output"] = model.recommend(
                    suggestion, selected_type, int(top_n)
                )
                st.rerun()

elif output:
    st.success(f"Animes similaires à **{output['query']}** ({kind})")

    for item in output["results"]:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                # Pas de HTML brut : Streamlit échappe le contenu, donc un titre
                # contenant < ou & ne casse pas la mise en page.
                st.subheader(item["name"], anchor=False)
                st.caption(f"{item['type']} · {item['genre']}")
            with right:
                note = "N/A" if item["rating"] is None else f"{item['rating']:.2f}/10"
                st.metric("Note", note)
                st.progress(
                    min(max(item["score"], 0.0), 1.0),
                    text=f"similarité {item['score']:.2f}",
                )
