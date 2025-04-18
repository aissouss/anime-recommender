import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AnimeRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["name", "genre"])
        self.df.reset_index(drop=True, inplace=True)
        self.df['name_clean'] = self.df['name'].str.lower().str.strip()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["genre"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, title, top_n=5):
        title = title.lower().strip()

        matches = self.df[self.df['name_clean'] == title]

        if matches.empty:
            suggestions = self.df[self.df['name_clean'].str.contains(title[:3])]
            if not suggestions.empty:
                possible = suggestions['name'].tolist()[:3]
                return f"Aucun anime trouvé. Vouliez-vous dire : {', '.join(possible)} ?"
            return f"Aucun anime trouvé avec le nom : {title}"

        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        results = [self.df.iloc[i]['name'] for i, _ in sim_scores]
        return results







