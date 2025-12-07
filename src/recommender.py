# src/recommender.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def _build_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Concatène les colonnes textuelles pour former une description
    sur laquelle on va entraîner TF-IDF ou BERT.
    """
    name = df["name"].fillna("")
    genre = df["genre"].fillna("")

    if "synopsis" in df.columns:
        synopsis = df["synopsis"].fillna("")
    else:
        synopsis = ""

    return (name + " " + genre + " " + synopsis).astype(str)


class BaseRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        # Nettoyage minimal
        self.df = self.df.dropna(subset=["name", "genre"])
        self.df["name"] = self.df["name"].astype(str).str.strip()
        self.df["name_clean"] = self.df["name"].str.lower()
        self.df["genre"] = self.df["genre"].astype(str)
        self.df["type"] = self.df.get("type", "Unknown").astype(str)
        self.df["rating"] = self.df.get("rating", 0).fillna(0)

        self.df.reset_index(drop=True, inplace=True)
        self.text = _build_text_column(self.df)

    def _find_index(self, title: str, type_filter: str | None = None):
        title_clean = title.lower().strip()

        df_filtered = self.df
        if type_filter and type_filter != "Tous":
            df_filtered = df_filtered[df_filtered["type"] == type_filter]

        matches = df_filtered[df_filtered["name_clean"] == title_clean]

        if matches.empty:
            # petite aide si l'utilisateur tape approximativement
            df_suggest = df_filtered[df_filtered["name_clean"].str.contains(title_clean[:3])]
            suggestions = df_suggest["name"].head(5).tolist()
            return None, suggestions

        idx = matches.index[0]
        return idx, None

    def available_types(self):
        types = sorted(self.df["type"].dropna().unique().tolist())
        return ["Tous"] + types


class TfidfRecommender(BaseRecommender):
    def __init__(self, csv_path: str):
        super().__init__(csv_path)

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.text)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, title: str, type_filter: str | None = None, top_n: int = 5):
        idx, suggestions = self._find_index(title, type_filter)

        if idx is None:
            if suggestions:
                return {
                    "error": f"Aucun anime trouvé pour « {title} ».",
                    "suggestions": suggestions,
                }
            return {"error": f"Aucun anime trouvé pour « {title} ».", "suggestions": []}

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        # On enlève l'anime lui-même
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]

        if type_filter and type_filter != "Tous":
            sim_scores = [
                (i, s) for i, s in sim_scores if self.df.loc[i, "type"] == type_filter
            ]

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

        results = []
        for i, score in sim_scores:
            row = self.df.loc[i]
            results.append(
                {
                    "name": row["name"],
                    "type": row["type"],
                    "genre": row["genre"],
                    "rating": float(row["rating"]),
                    "score": float(score),
                }
            )

        return {"results": results}


class EmbeddingRecommender(BaseRecommender):
    """
    Modèle plus avancé basé sur des embeddings BERT
    (sentence-transformers).
    """

    def __init__(self, csv_path: str, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(csv_path)

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers n'est pas installé. "
                "Ajoute-le dans requirements.txt et installe-le avec pip."
            )

        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.text.tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def recommend(self, title: str, type_filter: str | None = None, top_n: int = 5):
        idx, suggestions = self._find_index(title, type_filter)

        if idx is None:
            if suggestions:
                return {
                    "error": f"Aucun anime trouvé pour « {title} ».",
                    "suggestions": suggestions,
                }
            return {"error": f"Aucun anime trouvé pour « {title} ».", "suggestions": []}

        query_vec = self.embeddings[idx].reshape(1, -1)
        scores = (self.embeddings @ query_vec.T).ravel()

        indices = np.arange(len(self.df))
        mask = indices != idx

        if type_filter and type_filter != "Tous":
            mask = mask & (self.df["type"].values == type_filter)

        indices = indices[mask]
        scores = scores[mask]

        top_idx = indices[np.argsort(scores)[::-1][:top_n]]

        results = []
        for i in top_idx:
            row = self.df.loc[i]
            results.append(
                {
                    "name": row["name"],
                    "type": row["type"],
                    "genre": row["genre"],
                    "rating": float(row["rating"]),
                    "score": float(scores[np.where(indices == i)][0]),
                }
            )

        return {"results": results}







