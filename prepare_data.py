import pandas as pd

# Charger le dataset Kaggle dans le dossier data/
df = pd.read_csv("data/anime.csv")

# Garder les colonnes utiles selon TON dataset
df_clean = df[["Name", "Genres", "Type", "Score"]].copy()

# Renommer les colonnes pour simplifier le reste du projet
df_clean.rename(
    columns={
        "Name": "name",
        "Genres": "genre",
        "Type": "type",
        "Score": "rating",
    },
    inplace=True,
)

# Nettoyage
df_clean["name"] = df_clean["name"].astype(str).str.strip()
df_clean["genre"] = df_clean["genre"].fillna("").astype(str)
df_clean["type"] = df_clean["type"].fillna("Unknown").astype(str)
df_clean["rating"] = df_clean["rating"].fillna(0)

# Retirer les lignes inutiles
df_clean = df_clean[df_clean["name"] != ""]
df_clean = df_clean[df_clean["genre"] != ""]

# Sauvegarde
df_clean.to_csv("data/anime_clean.csv", index=False)

print("✔️ anime_clean.csv généré avec succès !")

