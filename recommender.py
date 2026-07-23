# src/recommender.py
"""Moteurs de recommandation basés contenu (TF-IDF et embeddings)."""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # dépendance optionnelle : l'app doit marcher sans
    SentenceTransformer = None

# Colonnes utilisées pour construire le texte à vectoriser.
# "synopsis" est optionnelle : si elle existe, elle est utilisée.
TEXT_COLUMNS = ("name", "genre", "synopsis")
ALL_TYPES = "Tous"


def _build_text_column(df: pd.DataFrame) -> pd.Series:
    """Concatène les colonnes textuelles disponibles en une seule description."""
    parts = [df[col].fillna("").astype(str) for col in TEXT_COLUMNS if col in df.columns]
    if not parts:
        raise ValueError(f"Aucune colonne textuelle parmi {TEXT_COLUMNS}")

    text = parts[0]
    for part in parts[1:]:
        text = text + " " + part
    return text.str.strip()


class BaseRecommender:
    """Charge et normalise les données. Les sous-classes fournissent la similarité.

    Patron de conception « template method » : `recommend()` est écrit une seule
    fois ici, chaque modèle n'implémente que `_similarities()`.
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        missing = {"name", "genre"} - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans {csv_path} : {sorted(missing)}")

        df = df.dropna(subset=["name", "genre"]).copy()
        df["name"] = df["name"].astype(str).str.strip()
        df["name_clean"] = df["name"].str.casefold()
        df["genre"] = df["genre"].astype(str)
        df["type"] = df["type"].astype(str) if "type" in df.columns else "Unknown"

        # Le dataset contient la chaîne "Unknown" dans Score : to_numeric la
        # transforme en NaN au lieu de faire planter float() plus tard.
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        else:
            df["rating"] = np.nan

        self.df = df.reset_index(drop=True)
        self.text = _build_text_column(self.df)
        self._aliases = self._build_alias_index()

    def _build_alias_index(self) -> pd.DataFrame:
        """Table (clé de recherche -> position) couvrant titre original ET anglais,
        pour qu'on puisse taper « Attack on Titan » comme « Shingeki no Kyojin »."""
        frames = [pd.DataFrame({"key": self.df["name_clean"], "pos": np.arange(len(self.df))})]

        if "name_en" in self.df.columns:
            english = self.df["name_en"].fillna("").astype(str).str.casefold().str.strip()
            extra = pd.DataFrame({"key": english, "pos": np.arange(len(self.df))})
            frames.append(extra[extra["key"] != ""])

        return pd.concat(frames, ignore_index=True).drop_duplicates(subset="key")

    # ------------------------------------------------------------------ #
    # Recherche du titre saisi par l'utilisateur
    # ------------------------------------------------------------------ #
    def _find_index(self, title: str) -> tuple[int | None, list[str]]:
        """Retourne (position, suggestions). La recherche ignore le filtre de
        type : on cherche l'anime source dans TOUT le catalogue, le filtre ne
        s'applique qu'aux résultats."""
        query = str(title).casefold().strip()
        if not query:
            return None, []

        exact = self._aliases.loc[self._aliases["key"] == query, "pos"]
        if len(exact) > 0:
            return int(exact.iloc[0]), []

        # regex=False : sinon un titre contenant ( ) + * fait planter la regex.
        partial = self._aliases["key"].str.contains(query, regex=False, na=False)
        if not partial.any():
            # Dernier recours : correspondance approximative (fautes de frappe,
            # romanisations différentes : "shippuden" vs "shippuuden").
            close = difflib.get_close_matches(query, self._aliases["key"], n=5, cutoff=0.6)
            partial = self._aliases["key"].isin(close)
            if not partial.any():
                return None, []

        positions = self._aliases.loc[partial, "pos"].unique()
        if len(positions) == 1:
            return int(positions[0]), []

        # On propose d'abord les mieux notés (plus probable d'être le bon).
        candidates = self.df.iloc[positions].sort_values("rating", ascending=False)
        return None, candidates["name"].head(5).tolist()

    def available_types(self) -> list[str]:
        return [ALL_TYPES] + sorted(self.df["type"].dropna().unique().tolist())

    # ------------------------------------------------------------------ #
    # À implémenter par chaque modèle
    # ------------------------------------------------------------------ #
    def _similarities(self, idx: int) -> np.ndarray:
        """Vecteur de similarité (taille n) entre l'item `idx` et tout le catalogue."""
        raise NotImplementedError

    def recommend(self, title: str, type_filter: str | None = None, top_n: int = 5) -> dict:
        idx, suggestions = self._find_index(title)
        if idx is None:
            return {
                "error": f"Aucun anime trouvé pour « {title} ».",
                "suggestions": suggestions,
            }

        scores = self._similarities(idx)

        mask = np.ones(len(self.df), dtype=bool)
        mask[idx] = False  # on n'auto-recommande pas l'anime lui-même
        if type_filter and type_filter != ALL_TYPES:
            mask &= self.df["type"].to_numpy() == type_filter

        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return {
                "error": f"Aucun anime de type « {type_filter} » à recommander.",
                "suggestions": [],
            }

        top_n = min(top_n, candidates.size)
        # argpartition : O(n) pour trouver les k meilleurs, au lieu d'un tri complet.
        best = candidates[np.argpartition(-scores[candidates], top_n - 1)[:top_n]]
        best = best[np.argsort(-scores[best])]  # tri des k seuls

        results = []
        for i in best:
            row = self.df.iloc[i]
            rating = row["rating"]
            results.append(
                {
                    "name": row["name"],
                    "type": row["type"],
                    "genre": row["genre"],
                    "rating": None if pd.isna(rating) else float(rating),
                    "score": float(scores[i]),
                }
            )

        return {"results": results, "query": self.df.at[idx, "name"]}


class TfidfRecommender(BaseRecommender):
    """Modèle rapide : sac de mots pondéré + similarité cosinus."""

    def __init__(self, csv_path: str, max_features: int = 50_000):
        super().__init__(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        self.matrix = self.vectorizer.fit_transform(self.text)

    def _similarities(self, idx: int) -> np.ndarray:
        # 1 ligne contre n, et non n x n : quelques Ko au lieu de 2,5 Go.
        return cosine_similarity(self.matrix[idx], self.matrix).ravel()


class EmbeddingRecommender(BaseRecommender):
    """Modèle sémantique : embeddings de phrases (sentence-transformers)."""

    def __init__(
        self,
        csv_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "data/cache",
    ):
        super().__init__(csv_path)

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers n'est pas installé : pip install sentence-transformers"
            )

        self.model_name = model_name
        self.embeddings = self._load_or_compute(Path(cache_dir))

    def _load_or_compute(self, cache_dir: Path) -> np.ndarray:
        """Calcule les embeddings une seule fois et les met en cache sur disque.

        La signature dépend du contenu + du modèle : si les données changent,
        le cache est automatiquement invalidé.
        """
        signature = hashlib.sha256(
            ("\n".join(self.text) + self.model_name).encode("utf-8")
        ).hexdigest()[:12]
        cache_file = cache_dir / f"embeddings_{signature}.npy"

        if cache_file.exists():
            return np.load(cache_file)

        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(
            self.text.tolist(),
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,  # -> produit scalaire = cosinus
            show_progress_bar=True,
        ).astype(np.float32)

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, embeddings)
        return embeddings

    def _similarities(self, idx: int) -> np.ndarray:
        return self.embeddings @ self.embeddings[idx]
