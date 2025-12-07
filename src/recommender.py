import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Nettoyage
        self.df.dropna(subset=["name", "genre", "type"], inplace=True)
        self.df['name_clean'] = self.df['name'].str.lower().str.strip()
        self.df.reset_index(drop=True, inplace=True)

        # TF-IDF sur genres
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['genre'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, title, type_, top_n=5):
        title = title.lower().strip()
        filtered_df = self.df[self.df["type"] == type_.lower()]
        
        matches = filtered_df[filtered_df["name_clean"] == title]

        if matches.empty:
            suggestions = filtered_df[filtered_df["name_clean"].str.contains(title[:3])]
            if not suggestions.empty:
                possible = suggestions["name"].tolist()[:3]
                return f"Aucun {type_} trouvé avec le nom : {title}\nVoulez-vous dire : {', '.join(possible)} ?"
            return f"Aucun {type_} trouvé avec le nom : {title}"

        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Filtrage uniquement par type
        sim_scores = [(i, score) for i, score in sim_scores if self.df.loc[i, "type"] == type_.lower() and i != idx]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

        results = [self.df.loc[i, "name"] for i, _ in sim_scores]
        return results







